import json
import pickle
import numpy as np

__locations = None
__data_columns = None
__model = None

def load_saved_artifacts():
    print('Started loading saved artifacts')
    global __locations
    global __data_columns
    global __model

    with open('./artifacts/columns.json', 'r') as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]

    if __model == None:
        with open('./artifacts/B_House_Price_Model.pickle', 'rb') as f:
            __model = pickle.load(f)

        print('Loading of artifacts is done')

def get_location_names():
    return __locations

def get_estimated_price(location, sqft, bhk, bath ):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
    pri = f"Rs {(round(__model.predict([x])[0], 2))} Lakhs"
    return pri

if __name__ == '__main__':
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price('2nd phase judicial layout', 1200, 4, 3))
    print(get_estimated_price('Random Location', 1200, 2, 3))
