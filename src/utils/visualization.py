# Guardar el modelo
model.save('captcha_solver_model.h5')

# Cargar el modelo
from tensorflow.keras.models import load_model
model = load_model('captcha_solver_model.h5')
