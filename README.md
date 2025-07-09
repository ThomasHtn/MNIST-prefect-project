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


### score :

[I 2025-07-09 12:01:07,471] Trial 0 finished with value: 0.9900833368301392 and parameters: {'lr': 0.003155037104389901, 'dropout': 0.13778711250083564}. Best is trial 0 with value: 0.9900833368301392.
[I 2025-07-09 12:02:07,672] Trial 1 finished with value: 0.9896666407585144 and parameters: {'lr': 0.003198053944448626, 'dropout': 0.464860622752001}. Best is trial 0 with value: 0.9900833368301392.
[I 2025-07-09 12:02:49,543] Trial 2 finished with value: 0.9852499961853027 and parameters: {'lr': 0.009774774222086769, 'dropout': 0.42061407725266897}. Best is trial 0 with value: 0.9900833368301392.
[I 2025-07-09 12:04:08,576] Trial 3 finished with value: 0.9897500276565552 and parameters: {'lr': 0.002515356238875022, 'dropout': 0.1842535387034419}. Best is trial 0 with value: 0.9900833368301392.
[I 2025-07-09 12:05:19,937] Trial 4 finished with value: 0.9898333549499512 and parameters: {'lr': 0.0005332474029560614, 'dropout': 0.3508606136594026}. Best is trial 0 with value: 0.9900833368301392.
[I 2025-07-09 12:07:49,156] Trial 5 finished with value: 0.9928333163261414 and parameters: {'lr': 0.0002481908599716149, 'dropout': 0.4026791618152127}. Best is trial 5 with value: 0.9928333163261414.
[I 2025-07-09 12:09:00,181] Trial 6 finished with value: 0.9908333420753479 and parameters: {'lr': 0.007524143817632057, 'dropout': 0.4494551834338226}. Best is trial 5 with value: 0.9928333163261414.
[I 2025-07-09 12:10:01,707] Trial 7 finished with value: 0.9887499809265137 and parameters: {'lr': 0.009277983917795508, 'dropout': 0.10474690351914817}. Best is trial 5 with value: 0.9928333163261414.
[I 2025-07-09 12:11:36,081] Trial 8 finished with value: 0.9901666641235352 and parameters: {'lr': 0.0014002008541583405, 'dropout': 0.4262832678873807}. Best is trial 5 with value: 0.9928333163261414.
[I 2025-07-09 12:13:06,890] Trial 9 finished with value: 0.9907500147819519 and parameters: {'lr': 0.0059538880914387715, 'dropout': 0.3949359885359637}. Best is trial 5 with value: 0.9928333163261414.
✅ Best hyperparameters: {'lr': 0.0002481908599716149, 'dropout': 0.4026791618152127}
🏆 Best validation accuracy: 0.9928
WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
✅ Final validation accuracy: 0.9918
💾 Model saved to models/latest_model.h5