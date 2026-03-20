#-------------------------------------------------------------------------------------
#                               SERVER CONFIGURATION
#
# TESTED on PYTHON : 3.11
# Works at PORT : 8000
# Required Dependencies : pip install fastapi , pip install scikit-learn
#-------------------------------------------------------------------------------------
#                                       USE
#
# 1. Keep EVALUATE.csv on ROOT of SERVER FOLDER
# 2. START SERVER 
# 3. POST FINAL.csv (each participant itetratively) to Endpoint : "/calculate-f1"
# 4. RETURNS A JSON OBJECT WITH ["F1 Score"] as the Evaluated Score
#-------------------------------------------------------------------------------------
#                      HOW TO POST REQUEST TO SERVER (TEST METHOD)
# 1. FOR TEST USE TEST CLIENT ON localhost:8000/
# 2. UPLOAD TEST.CSV in ROOT OF THIS FOLDER  (MATCH 30% FLIPED WITH CORRECT)
# 3. CLICK CALCULATE, ANSWER SHOULD BE (70.2987) 
# 2. UPLOAD EVALUATE.CSV in ROOT OF THIS FOLDER  (MATCH CORRECT WITH CORRECT)
# 3. CLICK CALCULATE, ANSWER SHOULD BE (100)
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
    file_name= "EVALUATE.csv"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "EVALUATE.csv")
    if os.path.exists(file_path):
        # Ensure column names match your CSV exactly (ID and CLASS)
        df = pd.read_csv(file_path)[['ID', 'CLASS']].drop_duplicates('ID')
        eval_data['df'] = df
        eval_data['df']['ID'] = eval_data['df']['ID'].astype(str)
        eval_data['df']['CLASS'] = eval_data['df']['CLASS'].astype(str)
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
async def calculate_f1(final_file: UploadFile = File(...)):
    if 'df' not in eval_data:
        raise HTTPException(status_code=500, detail="Server Error: EVALUATE.csv not loaded.")

    try:
        content = await final_file.read()
        participant_df = pd.read_csv(io.BytesIO(content))
        
        participant_df['ID'] = participant_df['ID'].astype(str)
        participant_df['CLASS'] = participant_df['CLASS'].astype(str)

        merged = pd.merge(
            eval_data['df'], 
            participant_df[['ID', 'CLASS']], 
            on='ID', 
            how='left', 
            suffixes=('_true', '_pred')
        )

        # FIX: The merged columns are named CLASS_true and CLASS_pred because you merged on ID
        merged['CLASS_pred'] = merged['CLASS_pred'].fillna("ID_MISSING_INVALID")

        score = f1_score(
            merged['CLASS_true'], 
            merged['CLASS_pred'], 
            average='weighted'
        )

        return {"f1_score": round(score, 6)*100}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Processing Error: {str(e)}")


# TEST CLIENT 
@app.get("/", response_class=HTMLResponse)
async def main_page():
    return """
    <html>
        <body style="font-family:sans-serif; text-align:center; padding-top:50px;">
            <h2>Upload Participant Results (FINAL.csv)</h2>
            <input type="file" id="csvFile" accept=".csv">
            <button onclick="upload()">Calculate Score</button>
            <h1 id="res" style="color:blue;"></h1>
            <script>
                async function upload() {
                    const resDiv = document.getElementById('res');
                    const file = document.getElementById('csvFile').files[0];
                    if(!file) return alert("Select a file!");
                    
                    const formData = new FormData();
                    formData.append("final_file", file);
                    
                    try {
                        const resp = await fetch('/calculate-f1', {method:'POST', body:formData});
                        const result = await resp.json();
                        
                        if (resp.ok) {
                            resDiv.innerText = "F1 Score: " + result.f1_score;
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