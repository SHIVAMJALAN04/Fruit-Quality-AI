FRUIT QUALITY AI — PROFESSIONAL STREAMLIT APP

FILES
- app.py: app source
- fruit_quality_model.keras: copy your existing trained model into this folder before running

SETUP
1. Put app.py and fruit_quality_model.keras in the same folder.
2. Open Command Prompt in that folder.
3. Run: python -m streamlit run app.py
4. Open http://localhost:8501

NOTES
- Create an account from the Create account tab, then sign in.
- Local user accounts and prediction history are stored in fruit_quality_users.db.
- The class label order in get_label_names() must match the label indices used in training.
  It currently assumes index 0 = Fresh and index 1 = Rotten. Verify against your notebook's
  dataset label names before relying on results.
- This is a local student demo. Do not expose it publicly without stronger production security.
