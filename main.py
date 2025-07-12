import pandas as pd
import io
from datetime import datetime, date

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse

# Import the scheduling logic from your other file
import scheduler_logic

app = FastAPI(
    title="Smart Scheduler API",
    description="An API to generate optimal caregiver schedules from an Excel file.",
    version="1.0.0",
)

# Define the columns expected in the output for consistency
OUTPUT_COLUMNS = [
    "Caregiver ID",
    "Caregiver Name",
    "Client ID",
    "Client Name",
    "Schedule Slot ID",
    "Shift Start",
    "Shift End",
    "Shift Duration (Hours)",
    "Compatibility Score",
    "Distance (km)",
]


@app.post("/generate-schedules/", tags=["Scheduling"])
async def create_schedules(
    start_date: date = Form(
        ..., description="The start date for filtering shifts (YYYY-MM-DD)."
    ),
    end_date: date = Form(
        ..., description="The end date for filtering shifts (YYYY-MM-DD)."
    ),
    num_schedules: int = Form(
        3, gt=0, le=10, description="Number of top alternative schedules to generate."
    ),
    file: UploadFile = File(
        ..., description="The Excel data file (must match the expected format)."
    ),
):
    """
    Uploads an Excel file, filters shifts within a date range, and returns an Excel file
    containing the top N optimal schedules.
    """
    if start_date > end_date:
        raise HTTPException(
            status_code=400, detail="Start date cannot be after end date."
        )

    # Read the uploaded file into an in-memory buffer
    # This avoids saving the file to disk on the server
    try:
        contents = await file.read()
        excel_buffer = io.BytesIO(contents)
    except Exception:
        raise HTTPException(
            status_code=400, detail="There was an error reading the uploaded file."
        )

    try:
        # --- 1. Load and Preprocess Data ---
        finder = scheduler_logic.FeasiblePairFinder(excel_buffer)
        finder.load_and_preprocess_data()

        # --- 2. Filter Shifts by Date Range (Crucial New Step) ---
        print(f"-> Initial number of shifts: {len(finder.clients_df)}")

        # Convert date objects to datetime for comparison
        start_dt = datetime.combine(start_date, datetime.min.time())
        end_dt = datetime.combine(end_date, datetime.max.time())

        # Filter the DataFrame
        original_clients_in_range_df = finder.clients_df[
            (finder.clients_df["start_dt"] >= start_dt)
            & (finder.clients_df["start_dt"] <= end_dt)
        ].copy()

        if original_clients_in_range_df.empty:
            raise HTTPException(
                status_code=404,
                detail="No open shifts found in the specified date range.",
            )

        finder.clients_df = original_clients_in_range_df
        print(
            f"-> Shifts after filtering for date range [{start_date} to {end_date}]: {len(finder.clients_df)}"
        )

        # --- 3. Run the Optimization Logic ---
        feasible_pairs = finder.find_all_feasible_pairs()

        if feasible_pairs.empty:
            raise HTTPException(
                status_code=404,
                detail="No feasible caregiver-shift pairs found for the given criteria and date range.",
            )

        assigner = scheduler_logic.OptimalAssigner(
            finder.caregivers_df, finder.clients_df, feasible_pairs
        )
        list_of_optimal_schedules = assigner.find_optimal_assignments(num_schedules)

        if not list_of_optimal_schedules:
            raise HTTPException(
                status_code=404,
                detail="The optimizer could not find any valid schedules. This might be due to overly restrictive constraints.",
            )

        # --- 4. Generate the Output Excel File in Memory ---
        output_buffer = io.BytesIO()
        with pd.ExcelWriter(output_buffer, engine="openpyxl") as writer:
            for i, schedule_df in enumerate(list_of_optimal_schedules):
                schedule_num = i + 1
                sheet_name_assigned = f"Optimal Schedule {schedule_num}"
                sheet_name_unassigned = f"Unassigned for Sched {schedule_num}"

                # Ensure consistent column order and format
                schedule_df.reindex(columns=OUTPUT_COLUMNS).to_excel(
                    writer, sheet_name=sheet_name_assigned, index=False
                )

                unassigned_shifts = original_clients_in_range_df[
                    ~original_clients_in_range_df["scheduleslotid"].isin(
                        schedule_df["Schedule Slot ID"]
                    )
                ]
                unassigned_shifts.to_excel(
                    writer, sheet_name=sheet_name_unassigned, index=False
                )

            # Write explanation sheet
            explanation_text = [
                f"File generated for shifts between {start_date.isoformat()} and {end_date.isoformat()}.",
                f"This file contains the top {len(list_of_optimal_schedules)} alternative schedules, sorted by total compatibility score.",
                "Each 'Optimal Schedule X' sheet contains one full, valid schedule for the date range.",
                "Each 'Unassigned for Sched X' sheet lists shifts in the date range not filled in that specific schedule.",
                f"Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ]
            explanation = pd.DataFrame({"Note": explanation_text})
            explanation.to_excel(writer, sheet_name="Explanation", index=False)

        output_buffer.seek(0)  # Rewind the buffer to the beginning

        # --- 5. Return the Excel file as a response ---
        filename = f"schedules_{start_date}_to_{end_date}.xlsx"
        return StreamingResponse(
            output_buffer,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )

    except (KeyError, ValueError) as e:
        # Catches data format errors (missing columns, bad dates, etc.)
        raise HTTPException(
            status_code=400,
            detail=f"Error processing the Excel file. Please check data format. Details: {e}",
        )
    except Exception as e:
        # Generic catch-all for unexpected errors from the scheduler logic
        raise HTTPException(
            status_code=500, detail=f"An unexpected internal error occurred: {e}"
        )
