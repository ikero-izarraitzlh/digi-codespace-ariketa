


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv('./data/players_22.csv')

# convert the long column-name line into a Python list
cols_line = """pace    shooting    passing    dribbling    defending    physic    attacking_crossing    attacking_finishing    attacking_heading_accuracy    attacking_short_passing    attacking_volleys    skill_dribbling    skill_curve    skill_fk_accuracy    skill_long_passing    skill_ball_control    movement_acceleration    movement_sprint_speed    movement_agility    movement_reactions    movement_balance    power_shot_power    power_jumping    power_stamina    power_strength    power_long_shots    mentality_aggression    mentality_interceptions    mentality_positioning    mentality_vision    mentality_penalties    mentality_composure    defending_marking_awareness    defending_standing_tackle    defending_sliding_tackle    goalkeeping_diving    goalkeeping_handling    goalkeeping_kicking    goalkeeping_positioning"""
column_names = cols_line.split()  # splits on any whitespace (tabs/spaces/newlines)
print(column_names)	


df = df.dropna(subset=['age', 'overall', 'potential', 'international_reputation', 'value_eur'] + column_names)

######################################################
##### HEMEN  ezaugarriak eta algoritmoa ALDATU #####

# aukeratu 6 ezaugarri:
ezaugarriak = ['age', 'overall', 'potential', 'international_reputation', 'pace', 'shooting']

# algoritmoa hautatu:
algoritmoa = 'GradientBoosting'  # 'LinearRegression', 'GradientBoosting', 'RandomForest'


######################################################
######################################################

asmatu_beharrekoa = 'value_eur'

results = []

X = df[ezaugarriak]
y = df[asmatu_beharrekoa]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

match algoritmoa:
    case 'LinearRegression':
        model = LinearRegression()
    case 'GradientBoosting':
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    case 'RandomForest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    case _:
        raise ValueError(f'Algoritmo ezezaguna: {algoritmoa}')
  

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)

# show some predictions vs actuals
# print(f'Features: {ezaugarriak}')
# for actual, pred in list(zip(y_test, preds))[:5]:
#   print(f'Actual: {actual:.2f}, Predicted: {pred:.2f}')  

mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

print(f'Mean Absolute Error: {mae:.2f} EUR, batazbeste zenbat okertu den balioan')
print(f'R^2 Score: {r2:.4f}, modeloa nola ondo doan azalduz (1.0 perfektua da)')  

