#-------------------------------------------------------------------------------------
#                              SERVER CONFIGURATION
#
# TESTED on PYTHON : 3.11
# Works at PORT : 8000
# Required Dependencies : pip install fastapi python-multipart pandas scikit-learn uvicorn
#-------------------------------------------------------------------------------------
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import pandas as pd
from sklearn.metrics import f1_score
import io
import os

#----------------------------- LIFESPAN HANDLER --------------------------------
eval_data = {}
@asynccontextmanager
async def lifespan(app: FastAPI):
    file_name = "EVALUATE.csv"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, file_name)
    
    if os.path.exists(file_path):
        # Read only the first two columns, regardless of their headers
        df = pd.read_csv(file_path).iloc[:, :2]
        
        # Standardize internal names so we never have to worry about headers again
        df.columns = ['IMAGE', 'LABEL_true']
        df = df.drop_duplicates('IMAGE')
        
        # Cast to string
        df['IMAGE'] = df['IMAGE'].astype(str)
        df['LABEL_true'] = df['LABEL_true'].astype(str)
        
        eval_data['df'] = df
        print(f"Successfully loaded {len(df)} records for evaluation.")
    else:
        print(f"CRITICAL ERROR: {file_path} not found.")
    
    yield
    eval_data.clear()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

#---------------------------------- END POINTS --------------------------------------
# CALCULATING F1 SCORE
@app.post("/calculate-f1")
async def calculate_f1(final_file: UploadFile = File(...)): # final_file accepts any file name
    if 'df' not in eval_data:
        raise HTTPException(status_code=500, detail="Server Error: EVALUATE.csv not loaded.")

    try:
        content = await final_file.read()
        participant_df = pd.read_csv(io.BytesIO(content))
        
        # Ensure the CSV actually has at least two columns to prevent index errors
        if len(participant_df.columns) < 2:
            raise HTTPException(status_code=400, detail="Uploaded CSV must have at least two columns.")

        # Strip everything except the first two columns
        participant_df = participant_df.iloc[:, :2]
        
        # Rename them to match our internal standard
        participant_df.columns = ['IMAGE', 'LABEL_pred']
        
        # Cast inputs to string
        participant_df['IMAGE'] = participant_df['IMAGE'].astype(str)
        participant_df['LABEL_pred'] = participant_df['LABEL_pred'].astype(str)

        # Merge based on our standardized 'IMAGE' column
        merged = pd.merge(
            eval_data['df'], 
            participant_df, 
            on='IMAGE', 
            how='left' 
        )

        # Handle missing or unmapped predictions
        merged['LABEL_pred'] = merged['LABEL_pred'].fillna("MISSING")

        # Calculate Multi-Class F1 Score
        score = f1_score(
            merged['LABEL_true'], 
            merged['LABEL_pred'], 
            average='weighted' 
        )

        return {"f1_score": round(score, 6) * 100}

    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="The uploaded CSV file is completely empty.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Processing Error: {str(e)}")

# TEST CLIENT 
@app.get("/", response_class=HTMLResponse)
async def main_page():
    return """
    <html>
        <body style="font-family:sans-serif; text-align:center; padding-top:50px;">
            <h2>Upload Participant Results</h2>
            <p style="color:gray; font-size:14px;">(Any file name works. Ensure Col 1 = IDs, Col 2 = Predictions)</p>
            <input type="file" id="csvFile" accept=".csv">
            <button onclick="upload()">Calculate Score</button>
            <h1 id="res" style="color:blue;"></h1>
            <script>
                async function upload() {
                    const resDiv = document.getElementById('res');
                    const file = document.getElementById('csvFile').files[0];
                    if(!file) return alert("Select a file!");
                    
                    const formData = new FormData();
                    formData.append("final_file", file); // Backend expects "final_file" form key, but doesn't care about the real file name.
                    
                    try {
                        const resp = await fetch('/calculate-f1', {method:'POST', body:formData});
                        const result = await resp.json();
                        
                        if (resp.ok) {
                            resDiv.innerText = "F1 Score: " + result.f1_score.toFixed(4);
                        } else {
                            resDiv.innerText = "Error: " + (result.detail || "Unknown error");
                        }
                    } catch (err) {
                        resDiv.innerText = "Network Error";
                    }
                }
            </script>
        </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)