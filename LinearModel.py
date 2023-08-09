from Calculations import *
from Load import *

def main():
    initial_w = 0.
    initial_b = 0.
    iterations = 1500
    alpha = 0.01
    x_train, y_train = load_data()

    w,b,_,_ = gradient_descent(x_train ,y_train, initial_w, initial_b, 
                        compute_cost, compute_gradient, alpha, iterations)
    pop, pred = prediction(w,b)
    print(f"For a population of {pop}, The predicted restaurant profit is ${pred}")

def prediction(w,b):
    population = float(input("Please enter a City Population: "))
    population /= 10000
    return population * 10000, (population * w + b) * 10000

if __name__ == "__main__":
    main()