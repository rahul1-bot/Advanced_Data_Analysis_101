#----------------------------------FUNCTION - 1 : REGRESSION METRIC ----------------------------------------------------------------------------------#
# Making the Metric function
def metric(y_test, y_preg):
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import median_absolute_error

    # Accessing the metrics
    mean_error = mean_squared_error(y_test, y_preg)
    root_mean_squared_error = np.sqrt(mean_error)
    score = r2_score(y_test, y_preg)
    absolute_error = mean_absolute_error(y_test, y_preg)
    median_error = median_absolute_error(y_test, y_preg)

    # Printing the Scores of various metrics
    print("Mean-Squared-Error : {}".format(mean_error))
    print("Root-Mean-Squared-Error : {}".format(root_mean_squared_error))
    print("Score : {}".format(score))
    print("Mean-Absolute-Error : {}".format(absolute_error))
    print("Median-Absolute-Error : {}".format(median_error))

    # Visualising the Result
    plt.figure(figsize=(12, 8))
    ax1 = sns.distplot(y_test, color="r", hist=False, label="Test Distribution")
    ax2 = sns.distplot(y_preg, color="b", hist=False, label="Predicted Distribution")
    plt.legend()
    plt.title("Test Vs Predicted Distribution")
    plt.show()

#---------------------------FUNCTION 2: TUNNED LINEAR REGRESSION ---------------------------------------------------#
# Setting the Linear Regression
def multiple_regression(x_train, x_test, y_train, y_test):
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import GridSearchCV
    regressor = LinearRegression()

    # Parameters to tune
    print("Parameters to be tune : {}".format(LinearRegression().get_params().keys()))

    # Setting and tuning the hyperparameters
    params = {"copy_X": [True, False],
              "fit_intercept": [True, False],
              "n_jobs": [-1],
              "normalize": [True, False]}

    grid = GridSearchCV(regressor, params, cv=5, scoring="r2")
    grid.fit(x_train, y_train)
    y_preg = grid.predict(x_test)

    # Displaying the best hyperparameters used for the result
    print("Best Hyperparameters Used : {}".format(grid.best_params_))

    # Setting the metric
    result = metric(y_test, y_preg)

# multiple_regression(x_train, x_test, y_train, y_test)

#-------------------------- FUNCTION-3 TUNNED SUPPORT VECTOR MACHINES -------------------------------------------------------#

# Setting the Support Vectors
def support_vectors_machine(x_train, x_test, y_train, y_test):
    from sklearn.svm import SVR
    from sklearn.model_selection import GridSearchCV
    regressor = SVR()

    # Hyper-Parameters to be tune:
    print("Hyper-Parameters to be tune : {}".format(SVR().get_params()))

    # Setting and tuning the hyperparameters
    params = {"C": [0.1, 1, 10, 100, 1000],
              "gamma": [1, 0.1, 0.01, 0.001],
              "kernel": ["rbf"],
              "epsilon": [0.1, 0.2, 0.5, 0.3]}

    grid = GridSearchCV(SVR(), params, cv=5, scoring="r2")
    grid.fit(x_train, y_train)
    y_preg = grid.predict(x_test)

    # Displaying the best hyperparametrs used for the model
    print("Best Hyper-Parameters used : {}".format(grid.best_params_))

    # Setting the metric
    result = metric(y_test, y_preg)

# support_vectors_machine(x_train, x_test, y_train, y_test)

#-----------------------FUNCTION-4: TUNNED DECISION TREES ----------------------------#
# Setting the Decision tree
def decision_tree(x_train, x_test, y_train, y_test):
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.model_selection import GridSearchCV

    regressor = DecisionTreeRegressor()

    # Parameters to be tune
    print("Hyper-Parameters to be tune : {}".format(DecisionTreeRegressor().get_params()))

    # Setting and tuning the hyperparameters
    params = {"criterion": ["mse", "mae"],
              "min_samples_split": [10, 20, 40],
              "max_depth": [2, 6, 8],
              "min_samples_leaf": [20, 40, 100],
              "max_leaf_nodes": [5, 20, 100]}

    grid = GridSearchCV(regressor, params, cv=5)
    grid.fit(x_train, y_train)
    y_preg = grid.predict(x_test)

    # Displaying the best hyperameters used
    print("Best Hyper-Parameters used : {}".format(grid.best_params_))

    # Setting the Metricc
    result = metric(y_test, y_preg)

# decision_tree(x_train, x_test, y_train, y_test)

#-----------------------------FUNCTION-5 TUNNED RANDOM FOREST -------------------------------------------------------#

# Setting the Random Forest
def random_forest(x_train, x_test, y_train, y_test):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV
    regressor = RandomForestRegressor()

    # Parameters to be tune
    print("Hyper-Parameters to be tune : {}".format(RandomForestRegressor().get_params().keys()))

    # Setting and tuning the hyperparameters
    params = {"n_estimators": [10, 50, 100],
              "max_features": ["auto", "log2", "sqrt"],
              "bootstrap": [True, False]}

    grid = GridSearchCV(regressor, params, cv=5)
    grid.fit(x_train, y_train)
    y_preg = grid.predict(x_test)

    # Displaying the best hyperameters
    print("Best Hyper-Parameters used : {}".format(grid.best_params_))

    # Setting the Metric
    result = metric(y_test, y_preg)

# random_forest(x_train, x_test, y_train, y_test)


#-------------------------------------FUNCTION-6 TUNNED XGBOOST------------------------------------------------------#
# Setting the XGBoost
def XGBoost(x_train, x_test, y_train, y_test):
    from xgboost import XGBRegressor
    from sklearn.model_selection import GridSearchCV
    import warnings
    warnings.filterwarnings("ignore")

    regressor = XGBRegressor()

    # Hyper-Params to be tune
    print("Hyper-parameters to be tune : {}".format(XGBRegressor().get_params().keys()))

    # Setting and tuning the hyperparameters
    params = {"nthread": [4],
              "learning_rate": [0.03, 0.05, 0.07],
              "max_depth": [5, 6, 7],
              "min_child_weight": [4],
              "subsample": [0.7],
              "colsample_bytree": [0.7],
              "n_estimators": [500]}

    grid = GridSearchCV(regressor, params, cv=5)
    grid.fit(x_train, y_train)
    y_preg = grid.predict(x_test)

    # displaying the best params used
    print("Best Hyper-Parameters used : {}".format(grid.best_params_))

    # Setting the metric
    result = metric(y_test, y_preg)

# XGBoost(x_train, x_test, y_train, y_test)


#---------------------------------FUNCTION-7: LASSO {FEATURE SELECTOR ALGORITHM} ---------------------------------------------#
def Lasso(x_train, y_train, x_test, y_test):
    from sklearn.linear_model import LassoCV

    # Setting the Hyper-parameters for Lasso
    lasso = LassoCV(alphas=[0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006,
                            0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1],
                    max_iter=50000,
                    cv=10)

    lasso.fit(x_train, y_train)

    # Printing the best alphas
    alpha = lasso.alpha_
    print("Best alpha :", alpha)

    # Tunnning the Alphas for more precision
    lasso = LassoCV(alphas=[alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8,
                            alpha * .85, alpha * .9, alpha * .95, alpha, alpha * 1.05,
                            alpha * 1.1, alpha * 1.15, alpha * 1.25, alpha * 1.3, alpha * 1.35,
                            alpha * 1.4],
                    max_iter=50000,
                    cv=10)

    lasso.fit(x_train, y_train)

    # Tunned Alphas
    alpha = lasso.alpha_
    print("Best alpha :", alpha)

    y_train_las = lasso.predict(x_train)
    y_test_las = lasso.predict(x_test)

    # Plotting the residuals
    plt.scatter(y_train_las, y_train_las - y_train, c="blue", marker="s", label="Training data")
    plt.scatter(y_test_las, y_test_las - y_test, c="lightgreen", marker="s", label="Validation data")
    plt.title("Linear regression with Lasso regularization")
    plt.xlabel("Predicted values")
    plt.ylabel("Residuals")
    plt.legend(loc="upper left")
    plt.hlines(y=0, xmin=10.5, xmax=13.5, color="red")
    plt.show()

    # Plot the predictions
    plt.scatter(y_train_las, y_train, c="blue", marker="s", label="Training data")
    plt.scatter(y_test_las, y_test, c="lightgreen", marker="s", label="Validation data")
    plt.title("Linear regression with Lasso regularization")
    plt.xlabel("Predicted values")
    plt.ylabel("Real values")
    plt.legend(loc="upper left")
    plt.plot([10.5, 13.5], [10.5, 13.5], c="red")
    plt.show()

    # Plotting the important coefficients
    coefs = pd.Series(lasso.coef_, index=x_train.columns)
    print("Lasso picked " + str(sum(coefs != 0)) + " features and eliminated the other " + \
          str(sum(coefs == 0)) + " features")
    imp_coefs = pd.concat([coefs.sort_values().head(10),
                           coefs.sort_values().tail(10)])
    imp_coefs.plot(kind="barh")
    plt.title("Coefficients in the Lasso Model")
    plt.show()

    # Setting the Metric
    print("-" * 40)
    result = metric(y_test, y_test_las)

# Lasso(x_train, y_train, x_test, y_test)

#-----------------------------FUNCTION - 8 ELASTICNET-------------------------------------------------#
def elasticNet(x_train, y_train, x_test, y_test):
    from sklearn.linear_model import ElasticNetCV

    # Setting the Hyper-parameters for ElasticNet
    elasticNet = ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1],
                              alphas=[0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006,
                                      0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6],
                              max_iter=50000, cv=10)

    elasticNet.fit(x_train, y_train)
    alpha = elasticNet.alpha_
    ratio = elasticNet.l1_ratio_
    print("Best l1_ratio :", ratio)
    print("Best alpha :", alpha)

    # Tunnnig the Hyper-parameters for more precision
    print("Increasing the precision with l1_ratio centered around " + str(ratio))
    elasticNet = ElasticNetCV(l1_ratio=[ratio * .85, ratio * .9, ratio * .95, ratio, ratio * 1.05,
                                        ratio * 1.1, ratio * 1.15],
                              alphas=[0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1,
                                      3, 6],
                              max_iter=50000, cv=10)

    elasticNet.fit(x_train, y_train)
    if (elasticNet.l1_ratio_ > 1):
        elasticNet.l1_ratio_ = 1

        # Printing the best alphas
    alpha = elasticNet.alpha_
    ratio = elasticNet.l1_ratio_
    print("Best l1_ratio :", ratio)
    print("Best alpha :", alpha)

    print("Tunning for more precision on alpha, with l1_ratio fixed at " + str(ratio) +
          " and alpha centered around " + str(alpha))

    elasticNet = ElasticNetCV(l1_ratio=ratio,
                              alphas=[alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85,
                                      alpha * .9,
                                      alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15, alpha * 1.25,
                                      alpha * 1.3,
                                      alpha * 1.35, alpha * 1.4],
                              max_iter=50000,
                              cv=10)

    elasticNet.fit(x_train, y_train)
    if (elasticNet.l1_ratio_ > 1):
        elasticNet.l1_ratio_ = 1

    alpha = elasticNet.alpha_
    ratio = elasticNet.l1_ratio_
    print("Best l1_ratio :", ratio)
    print("Best alpha :", alpha)

    y_train_ela = elasticNet.predict(x_train)
    y_test_ela = elasticNet.predict(x_test)

    # Plotting the residuals
    plt.scatter(y_train_ela, y_train_ela - y_train, c="blue", marker="s", label="Training data")
    plt.scatter(y_test_ela, y_test_ela - y_test, c="lightgreen", marker="s", label="Validation data")
    plt.title("Linear regression with ElasticNet regularization")
    plt.xlabel("Predicted values")
    plt.ylabel("Residuals")
    plt.legend(loc="upper left")
    plt.hlines(y=0, xmin=10.5, xmax=13.5, color="red")
    plt.show()

    # Plotting the predictions
    plt.scatter(y_train, y_train_ela, c="blue", marker="s", label="Training data")
    plt.scatter(y_test, y_test_ela, c="lightgreen", marker="s", label="Validation data")
    plt.title("Linear regression with ElasticNet regularization")
    plt.xlabel("Predicted values")
    plt.ylabel("Real values")
    plt.legend(loc="upper left")
    plt.plot([10.5, 13.5], [10.5, 13.5], c="red")
    plt.show()

    # Plotting the important coefficients
    coefs = pd.Series(elasticNet.coef_, index=x_train.columns)
    print("ElasticNet picked " + str(sum(coefs != 0)) + " features and eliminated the other " +
          str(sum(coefs == 0)) + " features")

    imp_coefs = pd.concat([coefs.sort_values().head(10),
                           coefs.sort_values().tail(10)])
    imp_coefs.plot(kind="barh")
    plt.title("Coefficients in the ElasticNet Model")
    plt.show()

    # Setting the Metric
    print("-" * 40)
    result = metric(y_test, y_test_ela)

# elasticNet(x_train, y_train, x_test, y_test)

#-------------------------------FUNCTION -9 RIDGE REGRESSION -----------------------------------------------#
def Ridge(x_train, y_train, x_test, y_test):
    from sklearn.linear_model import RidgeCV

    # Setting the Hyper-parameters for Ridge
    ridge = RidgeCV(alphas=[0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60])
    ridge.fit(x_train, y_train)
    alpha = ridge.alpha_
    print("Best alpha :", alpha)

    # Tunning the hyper-parameters for ridge
    print("Increasing the precision with alphas centered around " + str(alpha))
    ridge = RidgeCV(alphas=[alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85,
                            alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15,
                            alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4],
                    cv=10)
    ridge.fit(x_train, y_train)

    # Printing the best alphas
    alpha = ridge.alpha_
    print("Best alpha :", alpha)

    y_train_rdg = ridge.predict(x_train)
    y_test_rdg = ridge.predict(x_test)

    # Plotting the residuals
    plt.scatter(y_train_rdg, y_train_rdg - y_train, c="blue", marker="s", label="Training data")
    plt.scatter(y_test_rdg, y_test_rdg - y_test, c="lightgreen", marker="s", label="Validation data")
    plt.title("Linear regression with Ridge regularization")
    plt.xlabel("Predicted values")
    plt.ylabel("Residuals")
    plt.legend(loc="upper left")
    plt.hlines(y=0, xmin=10.5, xmax=13.5, color="red")
    plt.show()

    # Plotting the predictions
    plt.scatter(y_train_rdg, y_train, c="blue", marker="s", label="Training data")
    plt.scatter(y_test_rdg, y_test, c="lightgreen", marker="s", label="Validation data")
    plt.title("Linear regression with Ridge regularization")
    plt.xlabel("Predicted values")
    plt.ylabel("Real values")
    plt.legend(loc="upper left")
    plt.plot([10.5, 13.5], [10.5, 13.5], c="red")
    plt.show()

    # Plotting important coefficients
    coefs = pd.Series(ridge.coef_, index=x_train.columns)
    print("Ridge picked " + str(sum(coefs != 0)) + " features and eliminated the other "
          + str(sum(coefs == 0)) + " features")

    imp_coefs = pd.concat([coefs.sort_values().head(10),
                           coefs.sort_values().tail(10)])
    imp_coefs.plot(kind="barh")
    plt.title("Coefficients in the Ridge Model")
    plt.show()

    # Setting the Metric
    print("-" * 40)
    result = metric(y_test, y_test_rdg)


#Ridge(x_train, y_train, x_test, y_test)

#-----------------------------------------------------------------------------------------------------------------------#