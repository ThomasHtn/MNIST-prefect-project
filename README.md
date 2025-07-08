BAck : 
cd fastapi_app
uvicorn main:app --reload --host 0.0.0.0 --port 9500

Front : 
cd streamlit_app
streamlit run app.py

Prefect : 
From root folder
prefect server start

Run flow :
python3 -m prefect_flows.training_flow