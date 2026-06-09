from typing import Literal
import numpy as np
from sklearn.model_selection import train_test_split

from src.data_generation import DataGenerator
from src.lda import LDA
from src.qda import QDA
from src.nb import NB
from src.utils.plots import box_plot_a, box_plot_rho, scatter_plot, box_plot_real_data

class Experiments:
    def __init__(self):
        pass

    def experiment_a(
        self, 
        method_name: Literal["LDA", "QDA", "Naive Bayes"], 
        scheme: int, 
        rho=0.5, 
        a_arr=[0.1,0.5,1,2,3,5], 
        n=30, 
        pdf=None,
    ):
        a_arr = np.array(a_arr)

        results = {a: [] for a in a_arr}

        for a in a_arr:
            # data generation
            data_generator = DataGenerator()
            if scheme == 1:
                X, y = data_generator.scheme1(a)
            elif scheme == 2:
                X, y = data_generator.scheme2(a, rho=rho)

            for i in range(n):
                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=0.3,
                    train_size=0.7,
                )

                if method_name == "LDA":
                    method = LDA()
                elif method_name == "QDA":
                    method = QDA()
                elif method_name == "Naive Bayes":
                    method = NB()

                method.fit(X=X_train, y=y_train)

                y_pred = method.predict(Xtest=X_test)

                acc = (y_pred == y_test).mean()

                results[a].append(acc)

        box_plot_a(
            method_name=method_name,
            scheme=scheme,
            rho=rho,
            results=results,
            labels_name="a",
            pdf=pdf,
        )

    def experiment_rho(
        self, 
        method_name: Literal["LDA", "QDA", "Naive Bayes"], 
        scheme: int, 
        rho_arr=[0, 0.1, 0.3, 0.5, 0.7, 0.9], 
        a=2, 
        n=30, 
        pdf=None,
    ):
        rho_arr = np.array(rho_arr)

        results = {rho: [] for rho in rho_arr}

        for rho in rho_arr:
            # data generation
            data_generator = DataGenerator()
            if scheme == 1:
                X, y = data_generator.scheme1(a)
            elif scheme == 2:
                X, y = data_generator.scheme2(a, rho=rho)

            for i in range(n):
                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=0.3,
                    train_size=0.7,
                )

                if method_name == "LDA":
                    method = LDA()
                elif method_name == "QDA":
                    method = QDA()
                elif method_name == "Naive Bayes":
                    method = NB()

                method.fit(X=X_train, y=y_train)

                y_pred = method.predict(Xtest=X_test)

                acc = (y_pred == y_test).mean()

                results[rho].append(acc)

        box_plot_rho(
            method_name=method_name,
            scheme=scheme,
            a=a,
            results=results,
            labels_name="rho",
            pdf=pdf,
        )

    def experiment3(self, a=3, rho=0.4, pdf=None):
        data_generator = DataGenerator()
        X, y = data_generator.scheme2(a, rho=rho)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.3,
            train_size=0.7,
        )

        lda = LDA()
        lda.fit(X=X_train, y=y_train)

        qda = QDA()
        qda.fit(X=X_train, y=y_train)

        scatter_plot(
            X=X_train,
            y=y_train,
            a=a,
            rho=rho,
            lda=lda,
            qda=qda,
            pdf=pdf,
        )

    def experiment_real_data(
        self, 
        method_name: Literal["LDA", "QDA", "Naive Bayes"], 
        X, 
        y, 
        train_sizes=[0.9, 0.8, 0.7, 0.6, 0.5, 0.4], 
        test_sizes=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], 
        n=30,
        data=None,
        pdf=None,
    ):
        results = {train_size: [] for train_size in train_sizes}

        for train_size, test_size in zip(train_sizes, test_sizes):

            for i in range(n):

                X_train, X_test, y_train, y_test = train_test_split(
                    X, 
                    y,
                    test_size=test_size,
                    train_size=train_size,
                )

                if method_name == "LDA":
                    method = LDA(reg=1e-6)
                elif method_name == "QDA":
                    method = QDA(reg=1e-6)
                elif method_name == "Naive Bayes":
                    method = NB()

                method.fit(X=X_train, y=y_train)
                
                y_pred = method.predict(Xtest=X_test)

                acc = (y_pred == y_test).mean()

                results[train_size].append(acc)

        box_plot_real_data(
            method_name=method_name,
            results=results,
            labels_name="train size",
            data=data,
            pdf=pdf,
        )