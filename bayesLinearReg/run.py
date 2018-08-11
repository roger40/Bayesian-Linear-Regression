#!/usr/bin/env pytho      
# -*- coding: utf-8 -*-
import bayes


def main():
    """
    运行主函数
    :return:
    """
    xMat, yMat = bayes.dataLoad()
    trXM, trYM, teXM, teYM = bayes.dataDivide(xMat, yMat)
    print("Choose which way to use for prediction:\n1.Normal Linear Regression\n2.Bayesian Linear Regression")
    number = input()
    if number == "1":
        mu = bayes.normal(trXM, trYM)
        bayes.testBayes(mu, teXM, teYM)
    elif number == "2":
        mu = bayes.bayes(trXM, trYM)
        bayes.testBayes(mu, teXM, teYM)
    else:
        print("ERROR INPUT,TRY IT AGAIN")
        main()


if __name__ == "__main__":
    main()
