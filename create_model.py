from model import train_model
import pickle

model = train_model()
pickle.dump(model, open("model.pkl", "wb"))

print("Model saved successfully")