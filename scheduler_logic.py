import pandas as pd
import numpy as np
from geopy.distance import geodesic
from datetime import datetime, timedelta
import warnings
from typing import Set, List, Tuple, Dict, Any
from scipy.spatial import cKDTree
from ortools.sat.python import cp_model

warnings.filterwarnings("ignore")


class FeasiblePairFinder:
    """
    Finds all potentially feasible caregiver-client shift pairs.
    This class is responsible for filtering out pairs that violate static,
    unbreakable constraints (e.g., skills, distance, hard restrictions, availability).
    It does NOT handle dynamic constraints like hourly limits, as that will be
    managed globally by the optimizer.
    """

    ATTRIBUTE_MATCH_WEIGHT = 1.0
    # CHANGE MADE HERE: Set DISTANCE_BONUS_WEIGHT to 0.0 to remove distance bonus
    DISTANCE_BONUS_WEIGHT = 0.0
    SOFT_RESTRICTION_PENALTY = 0.5

    def __init__(self, excel_file_path: str):
        self.excel_file = excel_file_path
        self.caregivers_df = pd.DataFrame()
        self.clients_df = pd.DataFrame()
        self.restrictions_df = pd.DataFrame()
        self.attribute_mapping_df = pd.DataFrame()
        self.caregiver_schedules_df = pd.DataFrame()
        self.caregiver_tree: cKDTree | None = None
        self.caregivers_with_coords_indices = None
        self.max_travel_distance_km = 0.0
        self.client_need_to_caregiver_restriction_map = {
            "have cat(s)": "allergic /afraid of cat(s)",
            "have dog(s)": "allergic /afraid of dogs",
            "smoker": "allergic to smoke",
            "incontinent": "not ok with incontinence",
            "client is heavy and or needs transfers": "cannot do heavy transfers",
            "palliative": "no palliative experience",
        }

    def load_and_preprocess_data(self):
        """
        Loads data from Excel sheets and performs initial preprocessing
        including column standardization, parsing, and geospatial index setup.
        """
        print(f"-> Attempting to load data from: {self.excel_file}")
        try:
            self.caregivers_df = pd.read_excel(
                self.excel_file, sheet_name="Caregiver Details"
            )
            self.clients_df = pd.read_excel(self.excel_file, sheet_name="Open Shifts")
            self.restrictions_df = pd.read_excel(
                self.excel_file, sheet_name="Restriction Code"
            )
            self.attribute_mapping_df = pd.read_excel(
                self.excel_file, sheet_name="Attribute Code"
            )
            try:
                self.caregiver_schedules_df = pd.read_excel(
                    self.excel_file, sheet_name="Caregiver Schedules"
                )
            except Exception as e:
                print(
                    f"Warning: 'Caregiver Schedules' sheet not found or error loading: {e}. Proceeding without it."
                )
                self.caregiver_schedules_df = pd.DataFrame()

            # Standardize column names across all DataFrames
            all_dfs = [
                self.caregivers_df,
                self.clients_df,
                self.restrictions_df,
                self.attribute_mapping_df,
            ]
            if not self.caregiver_schedules_df.empty:
                all_dfs.append(self.caregiver_schedules_df)

            for df in all_dfs:
                df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

            # Filter for active caregivers
            self.caregivers_df = self.caregivers_df[
                ~self.caregivers_df["status"].isin(["inactive", "terminated"])
            ].copy()
            self.caregivers_df.reset_index(drop=True, inplace=True)
            self.clients_df.reset_index(drop=True, inplace=True)

            print(
                f"✓ Loaded and filtered to {len(self.caregivers_df)} active caregivers."
            )
            print(f"✓ Loaded {len(self.clients_df)} client shifts.")

            self._pre_parse_data()

            self.max_travel_distance_km = self.caregivers_df[
                "distancewillingtotravel"
            ].max()
            if (
                pd.isna(self.max_travel_distance_km)
                or self.max_travel_distance_km == np.inf
            ):
                self.max_travel_distance_km = 200

            self._setup_geospatial_index()

        except FileNotFoundError:
            raise FileNotFoundError(
                f"Error: The Excel file was not found at '{self.excel_file}'"
            )
        except KeyError as e:
            raise KeyError(
                f"Error: A required column is missing from your Excel file. Check headers. Missing: '{e}'"
            )
        except Exception as e:
            raise Exception(
                f"An unexpected error occurred while loading or preprocessing data: {e}. Check sheet names and file integrity."
            )

    def _pre_parse_data(self):
        """Helper to parse various string columns into sets/lists for efficiency."""
        print("-> Pre-parsing attributes, skills, restrictions, and availabilities...")

        for df in [self.restrictions_df, self.attribute_mapping_df]:
            for col in df.select_dtypes(include=["object"]).columns:
                df[col] = df[col].astype(str).str.strip().str.lower()

        # Always use 'scheduleslotid' as the normalized column name
        if (
            "scheduleslotid" not in self.clients_df.columns
            or self.clients_df["scheduleslotid"].isnull().all()
        ):
            print(
                "-> 'scheduleslotid' column not found or is empty in 'Open Shifts'. Generating unique IDs..."
            )
            self.clients_df["scheduleslotid"] = [
                f"Shift_{i + 1}" for i in range(len(self.clients_df))
            ]
        else:
            self.clients_df["scheduleslotid"] = self.clients_df[
                "scheduleslotid"
            ].astype(str)
            print("-> 'scheduleslotid' column found and populated. Using existing IDs.")

        self.caregivers_df["parsed_skills"] = self.caregivers_df["skilltype"].apply(
            self._parse_string_to_set
        )
        self.caregivers_df["parsed_attrs"] = self.caregivers_df[
            "attribute_categories"
        ].apply(self._parse_string_to_set)
        self.caregivers_df["parsed_restrictions"] = self.caregivers_df[
            "restriction_categories"
        ].apply(self._parse_string_to_set)
        self.caregivers_df["parsed_unavailability"] = self.caregivers_df[
            "caregiver_unavailability_details"
        ].apply(self._parse_unavailability)
        if "gender" in self.caregivers_df.columns:
            self.caregivers_df["gender"] = (
                self.caregivers_df["gender"].astype(str).str.strip().str.lower()
            )
        if "distancewillingtotravel" in self.caregivers_df.columns:
            self.caregivers_df["distancewillingtotravel"] = pd.to_numeric(
                self.caregivers_df["distancewillingtotravel"], errors="coerce"
            ).fillna(np.inf)
        else:
            self.caregivers_df["distancewillingtotravel"] = np.inf

        self.clients_df["parsed_skills"] = self.clients_df["skilltype"].apply(
            self._parse_string_to_set
        )
        self.clients_df["parsed_attrs"] = self.clients_df["attribute_categories"].apply(
            self._parse_string_to_set
        )
        self.clients_df["parsed_restrictions"] = self.clients_df[
            "restriction_categories"
        ].apply(self._parse_string_to_set)

        self.clients_df["start_dt"] = pd.to_datetime(
            self.clients_df["schedule_start_datetime"], errors="coerce"
        )
        self.clients_df["end_dt"] = pd.to_datetime(
            self.clients_df["schedule_end_datetime"], errors="coerce"
        )
        self.clients_df["duration_hours"] = (
            (self.clients_df["end_dt"] - self.clients_df["start_dt"]).dt.total_seconds()
            / 3600
        ).fillna(0)

        self.clients_df["duration_minutes"] = (
            self.clients_df["duration_hours"] * 60
        ).astype(int)

        self._parse_confirmed_schedules()

        print("✓ Pre-parsing complete.")

    def _parse_confirmed_schedules(self):
        """Pre-parses confirmed caregiver schedules from the caregiver_schedules_df."""
        self.caregivers_df["parsed_confirmed_schedules"] = [
            [] for _ in range(len(self.caregivers_df))
        ]

        if not self.caregiver_schedules_df.empty:
            print("-> Pre-parsing confirmed caregiver schedules...")
            confirmed_schedules_filtered = self.caregiver_schedules_df[
                self.caregiver_schedules_df["schedule_status"].isin(
                    ["scheduled", "approved"]
                )
            ].copy()

            confirmed_schedules_filtered["start_dt"] = pd.to_datetime(
                confirmed_schedules_filtered["schedule_start_datetime"], errors="coerce"
            )
            confirmed_schedules_filtered["end_dt"] = pd.to_datetime(
                confirmed_schedules_filtered["schedule_end_datetime"], errors="coerce"
            )
            confirmed_schedules_filtered.dropna(
                subset=["start_dt", "end_dt"], inplace=True
            )

            for cg_id, group in confirmed_schedules_filtered.groupby("caregiverid"):
                schedules = [(s, e) for s, e in zip(group["start_dt"], group["end_dt"])]
                cg_rows_to_update = self.caregivers_df["caregiverid"] == cg_id
                if cg_rows_to_update.any():
                    self.caregivers_df.loc[
                        cg_rows_to_update, "parsed_confirmed_schedules"
                    ] = [schedules] * cg_rows_to_update.sum()
                else:
                    print(
                        f"Warning: Caregiver ID {cg_id} from 'Caregiver Schedules' not found in 'Caregiver Details'."
                    )
            print("✓ Confirmed caregiver schedules pre-parsing complete.")
        else:
            print(
                "-> No 'Caregiver Schedules' data found or loaded. No confirmed schedules to check."
            )

    def _setup_geospatial_index(self):
        """Sets up the cKDTree for efficient spatial queries for caregivers."""
        print("-> Setting up geospatial index for caregivers...")
        caregivers_with_coords = self.caregivers_df.dropna(
            subset=["latitude", "longitude"]
        ).copy()

        if not caregivers_with_coords.empty:
            points = caregivers_with_coords[["latitude", "longitude"]].values
            self.caregiver_tree = cKDTree(points)
            self.caregivers_with_coords_indices = caregivers_with_coords.index.values
            print(f"✓ Geospatial index built for {len(points)} caregivers.")
        else:
            self.caregiver_tree = None
            self.caregivers_with_coords_indices = np.array([])
            print("-> No valid caregiver coordinates found to build geospatial index.")

    def _parse_string_to_set(self, attr_string: str) -> Set[str]:
        """Parses a pipe-separated string into a set of lowercased, stripped strings."""
        if pd.isna(attr_string) or not str(attr_string).strip():
            return set()
        return {
            attr.strip().lower()
            for attr in str(attr_string).split("||")
            if attr.strip()
        }

    def _parse_unavailability(
        self, unavail_str: str
    ) -> List[Tuple[datetime, datetime]]:
        """Parses unavailability strings into a list of datetime tuples."""
        if pd.isna(unavail_str) or not str(unavail_str).strip():
            return []
        slots = []
        for period in str(unavail_str).split("||"):
            try:
                start_str, end_str = period.split("<>")
                slots.append(
                    (pd.to_datetime(start_str.strip()), pd.to_datetime(end_str.strip()))
                )
            except (ValueError, TypeError):
                continue
        return slots

    def _check_time_conflict(
        self,
        shift_start: datetime,
        shift_end: datetime,
        schedules: List[Tuple[datetime, datetime]],
    ) -> bool:
        """Checks for overlap between a shift and a list of existing schedules."""
        for sched_start, sched_end in schedules:
            if shift_start < sched_end and sched_start < shift_end:
                return True
        return False

    def _calculate_distance(
        self, cg_lat: float, cg_lon: float, client_lat: float, client_lon: float
    ) -> float:
        """Calculates geodesic distance in kilometers between two lat/lon points."""
        if (
            pd.isna(cg_lat)
            or pd.isna(cg_lon)
            or pd.isna(client_lat)
            or pd.isna(client_lon)
        ):
            return np.inf
        try:
            return geodesic((cg_lat, cg_lon), (client_lat, client_lon)).kilometers
        except ValueError:
            return np.inf

    def _check_attribute_match(
        self, caregiver_attrs: Set[str], client_attrs: Set[str]
    ) -> int:
        """Counts how many client attributes match mapped caregiver attributes."""
        match_count = 0
        for client_attr in client_attrs:
            mapped_row = self.attribute_mapping_df[
                self.attribute_mapping_df["client_attribute"] == client_attr
            ]
            if not mapped_row.empty:
                cg_mapped_attr = mapped_row.iloc[0]["caregiver_attribute"]
                if cg_mapped_attr in caregiver_attrs:
                    match_count += 1
        return match_count

    def _check_restriction_violations(
        self, caregiver: pd.Series, client: pd.Series
    ) -> Tuple[bool, float]:
        """
        Checks for restriction violations (both hard and soft).
        Returns (is_hard_violation, total_soft_penalty).
        """
        is_hard_violation = False
        soft_penalty = 0.0

        client_parsed_restrictions = client["parsed_restrictions"]
        caregiver_parsed_restrictions = caregiver["parsed_restrictions"]
        caregiver_gender = caregiver.get("gender", "")

        if (
            "prefers female caregiver only" in client_parsed_restrictions
            and caregiver_gender == "male"
        ):
            restriction_info = self.restrictions_df[
                (self.restrictions_df["caregiver_attribute"] == "male caregiver")
                & (
                    self.restrictions_df["client_attribute"]
                    == "prefers female caregiver only"
                )
            ]
            if (
                not restriction_info.empty
                and restriction_info.iloc[0]["is_hard_restriction"] == "yes"
            ):
                is_hard_violation = True
            else:
                soft_penalty += self.SOFT_RESTRICTION_PENALTY

        if (
            "prefers male caregiver only" in client_parsed_restrictions
            and caregiver_gender == "female"
        ):
            restriction_info = self.restrictions_df[
                (self.restrictions_df["caregiver_attribute"] == "female caregiver")
                & (
                    self.restrictions_df["client_attribute"]
                    == "prefers male caregiver only"
                )
            ]
            if (
                not restriction_info.empty
                and restriction_info.iloc[0]["is_hard_restriction"] == "yes"
            ):
                is_hard_violation = True
            else:
                soft_penalty += self.SOFT_RESTRICTION_PENALTY

        for (
            client_need,
            cg_restriction,
        ) in self.client_need_to_caregiver_restriction_map.items():
            if (
                client_need in client["parsed_attrs"]
                and cg_restriction in caregiver_parsed_restrictions
            ):
                restriction_info = self.restrictions_df[
                    self.restrictions_df["caregiver_attribute"] == cg_restriction
                ]
                if (
                    not restriction_info.empty
                    and restriction_info.iloc[0]["is_hard_restriction"] == "yes"
                ):
                    is_hard_violation = True
                else:
                    soft_penalty += self.SOFT_RESTRICTION_PENALTY

        return is_hard_violation, soft_penalty

    def find_all_feasible_pairs(self) -> pd.DataFrame:
        """
        Finds all caregiver-shift pairs that satisfy static constraints
        (skills, distance, availability, hard restrictions).
        Hour limits are NOT checked here; they are handled by the global optimizer.
        """
        print("\n-> Finding all potentially feasible pairs (pre-optimization)...")
        results = []

        num_clients = len(self.clients_df)
        for cl_idx, client in self.clients_df.iterrows():
            print(
                f"\r   Processing shift {cl_idx + 1}/{num_clients} for potential matches...",
                end="",
            )

            if (
                self.caregiver_tree is None
                or pd.isna(client["schedule_start_address_latitude"])
                or pd.isna(client["schedule_start_address_longitude"])
            ):
                candidate_indices = self.caregivers_df.index
            else:
                radius_in_degrees = self.max_travel_distance_km / 111.0
                client_coords = [
                    client["schedule_start_address_latitude"],
                    client["schedule_start_address_longitude"],
                ]
                indices_in_tree = self.caregiver_tree.query_ball_point(
                    client_coords, r=radius_in_degrees
                )
                if len(indices_in_tree) > 0:
                    candidate_indices = self.caregivers_with_coords_indices[
                        indices_in_tree
                    ]
                else:
                    candidate_indices = []

            for cg_idx in candidate_indices:
                caregiver = self.caregivers_df.loc[cg_idx]

                # --- STATIC CONSTRAINT CHECKS ---
                distance = self._calculate_distance(
                    caregiver["latitude"],
                    caregiver["longitude"],
                    client["schedule_start_address_latitude"],
                    client["schedule_start_address_longitude"],
                )
                if distance > caregiver["distancewillingtotravel"]:
                    continue

                shift_start, shift_end = client["start_dt"], client["end_dt"]
                if (
                    pd.isna(shift_start)
                    or pd.isna(shift_end)
                    or client["duration_hours"] <= 0
                ):
                    continue

                if self._check_time_conflict(
                    shift_start, shift_end, caregiver["parsed_unavailability"]
                ):
                    continue
                if self._check_time_conflict(
                    shift_start, shift_end, caregiver["parsed_confirmed_schedules"]
                ):
                    continue

                if not client["parsed_skills"].issubset(caregiver["parsed_skills"]):
                    continue

                is_hard_violation, restriction_penalty = (
                    self._check_restriction_violations(caregiver, client)
                )
                if is_hard_violation:
                    continue

                # --- Calculate score for this POTENTIALLY feasible pair ---
                attr_score = self._check_attribute_match(
                    caregiver["parsed_attrs"], client["parsed_attrs"]
                )

                # dist_bonus calculation is kept, but its contribution will be zero due to DISTANCE_BONUS_WEIGHT = 0.0
                dist_bonus = 0.0
                willing_dist = caregiver["distancewillingtotravel"]
                if willing_dist > 0 and willing_dist != np.inf:
                    dist_bonus = max(0, (willing_dist - distance) / willing_dist)
                elif willing_dist == 0 and distance == 0:
                    dist_bonus = 1.0

                final_score = (
                    (attr_score * self.ATTRIBUTE_MATCH_WEIGHT)
                    + (
                        dist_bonus * self.DISTANCE_BONUS_WEIGHT
                    )  # This term will now be 0
                    - restriction_penalty
                )

                if final_score > 0:
                    results.append(
                        {
                            "caregiver_id": caregiver["caregiverid"],
                            "caregiver_name": caregiver.get("caregivername"),
                            "client_id": client.get("clientid"),
                            "client_name": client.get("clientname"),
                            "scheduleslotid": client.get("scheduleslotid"),
                            "start_dt": shift_start,
                            "end_dt": shift_end,
                            "duration_hours": client["duration_hours"],
                            "duration_minutes": client["duration_minutes"],
                            "score": final_score,
                            "distance_km": distance,
                        }
                    )

        print(f"\n✓ Found {len(results)} potential pairs for optimization.")
        return pd.DataFrame(results)


# --- NEW CLASS TO COLLECT TOP N SOLUTIONS ---
class TopNSolutionCollector(cp_model.CpSolverSolutionCallback):
    """
    A callback class to collect the top N best solutions found by the solver.
    """

    def __init__(self, pairs_df: pd.DataFrame, num_solutions_to_keep: int):
        super().__init__()
        self._pairs_df = pairs_df
        self._num_solutions_to_keep = num_solutions_to_keep
        self._solutions = []  # List to store (score, assigned_indices) tuples
        self._solution_count = 0

    def on_solution_callback(self):
        """Called by the solver for each new solution found."""
        current_score = self.ObjectiveValue()

        # Check for duplicates. The solver might report the same solution multiple times.
        if any(abs(s[0] - current_score) < 1e-6 for s in self._solutions):
            return  # Already have a solution with this score, skip.

        self._solution_count += 1

        assigned_indices = self._pairs_df.index[
            self._pairs_df.apply(lambda row: self.Value(row["var"]), axis=1) == 1
        ].tolist()

        self._solutions.append((current_score, assigned_indices))

        # Sort by score (descending) and keep only the top N
        self._solutions.sort(key=lambda x: x[0], reverse=True)
        self._solutions = self._solutions[: self._num_solutions_to_keep]

        # Optional: Print progress
        print(
            f"\r   -> Found solution #{self._solution_count} with score {current_score:.2f}. "
            f"Keeping top {len(self._solutions)}.",
            end="",
        )

    def get_solutions(self) -> List[Tuple[float, List[int]]]:
        """Returns the collected list of top solutions."""
        return self._solutions


class OptimalAssigner:
    """
    Takes a list of feasible pairs and uses a CP-SAT solver to find the optimal
    assignment that maximizes the total compatibility score while respecting
    all constraints (including hourly limits and non-overlapping shifts).

    Handles a special case for long shifts: a shift > 8 hours can be assigned,
    but the caregiver cannot take any other shifts on that same day.
    """

    def __init__(
        self,
        caregivers_df: pd.DataFrame,
        shifts_df: pd.DataFrame,
        feasible_pairs_df: pd.DataFrame,
    ):
        self.caregivers = caregivers_df
        self.shifts = shifts_df
        self.pairs = feasible_pairs_df

        self.MAX_DAILY_WORK_MINUTES = 8 * 60
        self.MAX_WEEKLY_WORK_MINUTES = 40 * 60

    def find_optimal_assignments(self, num_schedules: int) -> List[pd.DataFrame]:
        if self.pairs.empty:
            print("No feasible pairs found. Cannot run optimization.")
            return []

        print(
            f"\n-> Building optimization model to find top {num_schedules} schedules..."
        )
        model = cp_model.CpModel()

        # Create a boolean variable for each potential assignment
        self.pairs["var"] = self.pairs.apply(
            lambda row: model.NewBoolVar(
                f"assign_{row['caregiver_id']}_to_{row['scheduleslotid']}"
            ),
            axis=1,
        )

        # Constraint: Each shift can be assigned to at most one caregiver
        for shift_id, group in self.pairs.groupby("scheduleslotid"):
            model.Add(sum(group["var"]) <= 1)

        # Constraints per caregiver
        for cg_id, cg_pairs in self.pairs.groupby("caregiver_id"):
            # Constraint: A caregiver cannot be assigned overlapping shifts
            for i in range(len(cg_pairs)):
                for j in range(i + 1, len(cg_pairs)):
                    pair1 = cg_pairs.iloc[i]
                    pair2 = cg_pairs.iloc[j]
                    # Check for time overlap
                    if max(pair1["start_dt"], pair2["start_dt"]) < min(
                        pair1["end_dt"], pair2["end_dt"]
                    ):
                        model.Add(pair1["var"] + pair2["var"] <= 1)

            # Daily hour limits logic
            cg_pairs["day"] = cg_pairs["start_dt"].dt.date
            for day, day_group in cg_pairs.groupby("day"):
                long_shifts = day_group[
                    day_group["duration_minutes"] > self.MAX_DAILY_WORK_MINUTES
                ]
                normal_shifts = day_group[
                    day_group["duration_minutes"] <= self.MAX_DAILY_WORK_MINUTES
                ]

                if not normal_shifts.empty:
                    daily_minutes_normal = sum(
                        normal_shifts["var"] * normal_shifts["duration_minutes"]
                    )
                    model.Add(daily_minutes_normal <= self.MAX_DAILY_WORK_MINUTES)

                if not long_shifts.empty:
                    long_shift_vars = long_shifts["var"].tolist()
                    model.Add(sum(long_shift_vars) <= 1)

                    is_long_shift_taken = model.NewBoolVar(
                        f"cg_{cg_id}_day_{day}_long_shift_taken"
                    )
                    model.Add(sum(long_shift_vars) >= 1).OnlyEnforceIf(
                        is_long_shift_taken
                    )
                    model.Add(sum(long_shift_vars) == 0).OnlyEnforceIf(
                        is_long_shift_taken.Not()
                    )

                    if not normal_shifts.empty:
                        model.Add(sum(normal_shifts["var"]) == 0).OnlyEnforceIf(
                            is_long_shift_taken
                        )

            # Weekly hour limits
            cg_pairs["week"] = (
                cg_pairs["start_dt"]
                .dt.to_period("W")
                .apply(lambda p: p.start_time)
                .dt.date
            )
            for week, week_group in cg_pairs.groupby("week"):
                weekly_minutes = sum(week_group["var"] * week_group["duration_minutes"])
                model.Add(weekly_minutes <= self.MAX_WEEKLY_WORK_MINUTES)

        # Objective function: Maximize the sum of scores
        SCALE_FACTOR = 100
        self.pairs["scaled_score"] = (self.pairs["score"] * SCALE_FACTOR).astype(int)

        total_score = sum(self.pairs["var"] * self.pairs["scaled_score"])
        model.Maximize(total_score)

        # --- MODIFIED SOLVING PROCESS ---
        print("-> Solving the optimization problem (this may take a moment)...")
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 60.0

        # Use the solution collector
        solution_collector = TopNSolutionCollector(self.pairs, num_schedules)
        status = solver.Solve(model, solution_collector)
        print()  # Newline after the progress indicator

        top_solutions = solution_collector.get_solutions()

        if not top_solutions:
            print("❌ No feasible solution found.")
            if status == cp_model.INFEASIBLE:
                print(
                    "   The problem is INFEASIBLE. No solution satisfies all hard constraints."
                )
            return []

        print(
            f"✓ Solver finished. Found {len(top_solutions)} unique, high-scoring schedules."
        )

        # Process the collected solutions into a list of DataFrames
        all_schedules_dfs = []
        for i, (score, assigned_indices) in enumerate(top_solutions):
            print(f"   - Schedule #{i + 1} Total Score: {score / SCALE_FACTOR:.2f}")
            results_df = self.pairs.loc[assigned_indices].copy()

            final_df = results_df[
                [
                    "caregiver_id",
                    "caregiver_name",
                    "client_id",
                    "client_name",
                    "scheduleslotid",
                    "start_dt",
                    "end_dt",
                    "duration_hours",
                    "score",
                    "distance_km",
                ]
            ].rename(
                columns={
                    "caregiver_id": "Caregiver ID",
                    "caregiver_name": "Caregiver Name",
                    "client_id": "Client ID",
                    "client_name": "Client Name",
                    "scheduleslotid": "Schedule Slot ID",
                    "start_dt": "Shift Start",
                    "end_dt": "Shift End",
                    "duration_hours": "Shift Duration (Hours)",
                    "score": "Compatibility Score",
                    "distance_km": "Distance (km)",
                }
            )

            final_df["Compatibility Score"] = final_df["Compatibility Score"].round(2)
            final_df["Distance (km)"] = final_df["Distance (km)"].round(2)
            final_df["Shift Start"] = final_df["Shift Start"].dt.strftime(
                "%Y-%m-%d %H:%M"
            )
            final_df["Shift End"] = final_df["Shift End"].dt.strftime("%Y-%m-%d %H:%M")

            all_schedules_dfs.append(
                final_df.sort_values(by="Shift Start").reset_index(drop=True)
            )

        return all_schedules_dfs


if __name__ == "__main__":
    excel_file_path = "SmartSchedular_Dataset_HospallAgency (2).xlsx"
    output_file = "top_n_optimal_assignments.xlsx"
    num_schedules_to_generate = 3  # <-- SET THE NUMBER OF SCHEDULES YOU WANT HERE

    try:
        finder = FeasiblePairFinder(excel_file_path)
        finder.load_and_preprocess_data()
        feasible_pairs = finder.find_all_feasible_pairs()

        if not feasible_pairs.empty:
            assigner = OptimalAssigner(
                finder.caregivers_df, finder.clients_df, feasible_pairs
            )
            list_of_optimal_schedules = assigner.find_optimal_assignments(
                num_schedules_to_generate
            )

            if list_of_optimal_schedules:
                print(f"\n📊 TOP {len(list_of_optimal_schedules)} SCHEDULES SUMMARY 📊")
                print("-" * 40)

                with pd.ExcelWriter(output_file) as writer:
                    for i, schedule_df in enumerate(list_of_optimal_schedules):
                        schedule_num = i + 1
                        sheet_name_assigned = f"Optimal Schedule {schedule_num}"
                        sheet_name_unassigned = (
                            f"Unassigned for Schedule {schedule_num}"
                        )

                        # Write the assigned shifts for this schedule
                        schedule_df.to_excel(
                            writer, sheet_name=sheet_name_assigned, index=False
                        )

                        # Calculate and write the unassigned shifts for this specific schedule
                        unassigned_shifts = finder.clients_df[
                            ~finder.clients_df["scheduleslotid"].isin(
                                schedule_df["Schedule Slot ID"]
                            )
                        ]
                        unassigned_shifts.to_excel(
                            writer, sheet_name=sheet_name_unassigned, index=False
                        )

                        # Print summary to console
                        total_shifts = len(finder.clients_df)
                        assigned_shifts = len(schedule_df)
                        avg_score = schedule_df["Compatibility Score"].mean()
                        total_score = schedule_df["Compatibility Score"].sum()

                        print(f"SCHEDULE #{schedule_num}:")
                        print(f"  - Total Score: {total_score:.2f}")
                        print(
                            f"  - Shifts Assigned: {assigned_shifts}/{total_shifts} ({assigned_shifts / total_shifts:.1%})"
                        )
                        print(f"  - Average Score: {avg_score:.2f}")
                        print(f"  - Saved to sheet: '{sheet_name_assigned}'")
                        print("-" * 40)

                    # Write a final explanation sheet
                    explanation_text = [
                        f"This file contains the top {len(list_of_optimal_schedules)} alternative schedules, sorted by the highest total compatibility score.",
                        "Each 'Optimal Schedule X' sheet contains one full, valid schedule.",
                        "Each 'Unassigned for Schedule X' sheet lists shifts that could not be filled in that specific schedule.",
                        "All schedules respect hard constraints: time conflicts, daily/weekly hour limits, skills, distance, and restrictions.",
                        f"Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    ]
                    explanation = pd.DataFrame({"Note": explanation_text})
                    explanation.to_excel(writer, sheet_name="Explanation", index=False)

                print(
                    f"\n✅ Process completed! Top {len(list_of_optimal_schedules)} schedules saved to: {output_file}"
                )

            else:
                print("\nNo feasible schedules could be generated by the optimizer.")
        else:
            print(
                "\nNo feasible pairs were found, so no assignments could be made by the optimizer."
            )

    except (FileNotFoundError, IOError) as e:
        print(f"\n❌ ERROR: A file loading error occurred: {e}")
        print(
            "Please ensure 'SmartSchedular_Dataset_HospallAgency (2).xlsx' is in the same directory as the script."
        )
    except KeyError as e:
        print(f"\n❌ ERROR: A required column is missing from your Excel data: {e}")
        print(
            "Please check your Excel sheet headers match the expected column names in the code (e.g., 'caregiverid', 'skilltype', 'latitude', 'schedule_start_datetime', etc.)."
        )
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
