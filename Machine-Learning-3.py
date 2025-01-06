import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Example dataset creation for illustration purposes
data = pd.DataFrame({
    'SquareFootage': [1500, 2500, 1800, 2200],
    'NumberOfBedrooms': [3, 4, 3, 4],
    'Price': [300000, 450000, 350000, 400000]
})

# Define features and target
X = data[['SquareFootage', 'NumberOfBedrooms']]  # Features (independent variables)
y = data['Price']  # Target variable (dependent variable)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")

print(f"Predictions: {predictions}")
print(f"Actual Values: {y_test.values}")

# Predict the price of a new flat based on user input
try:
    square_footage = float(input("Enter the square footage of the flat: "))
    number_of_bedrooms = int(input("Enter the number of bedrooms: "))

    new_flat = pd.DataFrame({
        'SquareFootage': [square_footage],
        'NumberOfBedrooms': [number_of_bedrooms]
    })
    new_flat_price = model.predict(new_flat)
    print(
        f"Predicted price for the new flat ({square_footage} sq ft, {number_of_bedrooms} bedrooms): ${new_flat_price[0]:,.2f}")
except ValueError:
    print("Invalid input. Please enter numerical values for square footage and number of bedrooms.")

