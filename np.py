import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from math import sqrt
from math import floor
from matplotlib.finance import candlestick_ohlc
import datetime

#Parameters brownian()
#
#x[0]:      X(0), initial stock price
#N:         number of increments
#T:         time period
#dt:        time step
#delta:     "speed" of the Brownian motion, variance = delta**2t
N = 150
T = 75
dt = T/N
delta = .1
delta2 = .05
x = np.empty((N+1))
xL = np.empty((N+1))
xH = np.empty((N+1))
dxL = np.empty((N+1))
dxH = np.empty((N+1))
x[0] = 10
xL[0] = 0.0
xH[0] = xL[0]
t = np.arange(0, T, dt)
portfolio = np.zeros(shape=(N,2))

def brownian(x0, n, dt, delta, out=None):
    x0 = np.asarray(x0)
    r = norm.rvs(size=x0.shape + (n,), scale=delta*sqrt(dt))
    if out is None:
        out = np.empty(r.shape)
    np.cumsum(r, axis=-1, out=out)
    out += np.expand_dims(x0, axis=-1)
    return out

def brownianu(s, N):
    np.random.seed(s)
    return np.cumsum(np.random.normal(0., 1., N)*sqrt(1./N))

#Parameters GBM()
#
#x[0]:      X(0), initial stock price (used by brownian())
#mu:        drift coefficient
#sigma:     volatility coefficient
#x:         brownian motion calculated by brownian()
#t:         np array, 0, dt, 2dt, ..., T
mu = 0.05
sigma = 0.04

def GBM(x0, mu, sigma, x, T, N):
    t = np.linspace(0., 1., N+1)
    S = []
    S.append(x0)
    for i in range(1, N+1):
        dr = (mu - 0.5 * sigma**2)*t[i]
        diff = sigma * x[i-1]
        S.append(x0*np.exp(dr+diff))
    return S, t

#Parameters stochrsi()
#
#x:         numpy array, prices (brownian, GBM, real datas)
#i:         stochrsi() computes the stochrsi of x at i
#stochrsit: stochrsi based on the last stochrsit prices, condition : stochrsit=n*dt, where n is a natural number and stochrsit is a natural number
stochrsit = 20
portfolio[stochrsit,1] = 0 #initial condition (qty of money 1)
for i in range(0, stochrsit+1) :
    portfolio[i,0] = 100 #initial condition (qty of money 0)
srsifiltered = np.empty((N-stochrsit))

def stochrsi(x, i, stochrsit):
    xH = np.amax(x[i-stochrsit:i])
    xL = np.amin(x[i-stochrsit:i])
    return (x[i-1]-xL)/(xH-xL)

#Paramters stochrsib()
#
#srsi:  numpy array of the values calculated by stochrsi()
def stochrsib(srsi):
    for i in range(0, len(srsi)):
        if(srsi[i]<0.5): #treshold
            srsifiltered[i] = 0
        elif(srsi[i]>=0.7): #treshold
            srsifiltered[i] = 1
        else:
            srsifiltered[i]=srsi[i]
    return srsifiltered

#Parameters ExpAv()
#
#alpha:     smoothing constant
#x0:        initial condition
#x:         set of datas
expava = np.empty((N+1))

def ExpAv(alpha, x0, x):
    expava[0] = x0
    for i in range(1,len(x)):
        expava[i] = alpha*x[i] + (1-alpha)*expava[i-1]
    return np.delete(expava, 1, axis = 0)

def buy(x, t, q):
    return [-q],[q/x[t]]

def sell(x, t, q):
    return [q*x[t]],[-q]

def plotexpav(t, expava, name):
    fig, ax = plt.subplots()
    plt.plot(t, expava)
    plt.savefig(name+'.png')

def plotsrsi(t, srsi, stochrsit, name):
    fig, ax = plt.subplots()
    srsi = np.delete([tuple([stochrsi(x, i, stochrsit)]) for i in range(stochrsit,len(t)+1)], 1, axis=0)
    plt.plot(np.arange(stochrsit*dt, T, dt), srsi)
    plt.savefig(name+'.png')

def plotportfolio(i, portfolio, lsrsi, name, x):
    fig, ax = plt.subplots()
    tplub = np.arange(stochrsit, lsrsi+stochrsit)
    for v in range(len(tplub)):
        tplub[v] = tplub[v]/dt
    if i==1:
        p = np.delete([tuple([portfolio[i, 0]+x[i]*portfolio[i, 1]]) for i in range((len(tplub)+1))], 1, axis=0)
        plt.plot(tplub, p)
    else:
        p = np.delete([tuple([portfolio[i, 0]/x[i]+portfolio[i, 1]]) for i in range((len(tplub)+1))], 1, axis=0)
        plt.plot(tplub, p)
    plt.savefig(name+'.png')

def plotsrsifiltered(t, srsifiltered, stochrsit, name):
    fig, ax = plt.subplots()
    plt.plot(np.arange(stochrsit*dt, T, dt), srsifiltered)
    plt.savefig(name+'.png')

def plotbrownian(x, dxH, dxL, dt, t, name, b):
    dxL = -abs(dxL)
    dxH = abs(dxH)
    for i in range(len(x)):
        dxL[i] *= (x[i]-x[i-1])/x[i]
        dxH[i] *= (x[i]-x[i-1])/x[i]
    quotes = [tuple([t[i],
                     x[i],
                     x[i+1]+dxH[i+1],
                     x[i+1]+dxL[i+1],
                     x[i+1]]) for i in range(b,len(x)-1)]
    fig, ax = plt.subplots()
    candlestick_ohlc(ax, quotes, width=dt, colorup='g', colordown='r', alpha=1.0)
    plt.savefig(name + '.png')

def stochrsiarray(x, t, stochrsit):
    return np.delete([tuple([stochrsi(x, i, stochrsit)]) for i in range(stochrsit,len(t)+1)], 1, axis=0)

"""le choix de treshold dépendra des risques de l'action : risque qu'on souhaite encourir (à voir si je définis ça par rapport à un gain min ou pas seulement) (si élevé alors en vente : treshold très important, on hold plus, en vente : treshold élevé, on hold plus, &inversement), à réfléchir si on all-in à chaque fois ou si l'on met en jeu f(srsi)*capital"""

def method1(stochrsit, t, x, portfolio, srsib, srsi, dt, p):
    for i in range(stochrsit, len(srsi)+stochrsit-1):
        if portfolio[i, 0] != 0 and srsib[i-stochrsit] == 0 :
            portfolio[i+1,0] = portfolio[i, 0] + buy(x, i+1, (1-srsi[i-stochrsit])**p*portfolio[i,0])[0]
            portfolio[i+1,1] = portfolio[i, 1] + buy(x, i+1, (1-srsi[i-stochrsit])**p*portfolio[i,0])[1]
            with open("run.txt", "a") as myfile:
                myfile.write("portfolio[][]B : " + str(portfolio[i,0]) + "," + str(portfolio[i,1]) + " -> " + str(portfolio[i+1,0])+","+str(portfolio[i+1,1]) + "*" + str(x[i+1]) + "\n")
        elif portfolio[i, 1] != 0 and srsib[i-stochrsit] == 1:
            portfolio[i+1,0] = portfolio[i,0] + sell(x, i+1, (srsi[i-stochrsit]*portfolio[i, 1])**p)[0]
            portfolio[i+1,1] = portfolio[i,1] + sell(x, i+1, (srsi[i-stochrsit]*portfolio[i, 1])**p)[1]
            with open("run.txt", "a") as myfile:
                myfile.write("portfolio[][]S : " + str(portfolio[i, 0]) + "," + str(portfolio[i, 1]) + " -> " + str(portfolio[i+1,0]) + "," + str(portfolio[i+1,1]) + "*" + str(x[i+1])+ "\n")
        else :
            portfolio[i+1, 0] = portfolio[i, 0]
            portfolio[i+1, 1] = portfolio[i, 1]
            with open("run.txt", "a") as myfile:
                myfile.write("portfolio[][]K : " + str(portfolio[i, 0]) + "," + str(portfolio[i, 1])+ " -> " + str(portfolio[i+1,0]) + "," + str(portfolio[i+1,1])+"*"+str(x[i+1])+"\n")
    return portfolio

def run(x, N, dt, delta, dxL, delta2, dxH, t, i, portfolio, p):
    brownian(x[0], N, dt, delta, out=x[1:])
    '''brownian(dxL[0], N, dt, delta2, out=dxL[1:])
    brownian(dxH[0], N, dt, delta2, out=dxH[1:])'''
    plotbrownian(x, dxH*0, dxL*0, dt, t, 'plotb'+str(i), 0)
    xgbm = GBM(x[0], mu, sigma, x, T, N)[0]
    plotbrownian(xgbm, dxH*0, dxL*0, dt, t, 'plotgbm'+str(i), 1)
    srsi = stochrsiarray(x, t, stochrsit)
    srsib = stochrsib(srsi)
    plotsrsifiltered(t, srsib, stochrsit, 'plotsrsifiltered'+str(i))
    plotsrsi(t, srsi, stochrsit, 'plotsrsi'+str(i))
    portfolio = method1(stochrsit, t, x, portfolio, srsib, srsi, dt, p)
    plotportfolio(1, portfolio, len(srsi), 'plotportfolio1b'+str(i), x)
    plotportfolio(0, portfolio, len(srsi), 'plotportfolio0b'+str(i), x)
    portfolio2 = method1(stochrsit, t, xgbm, portfolio, srsib, srsi, dt, p)
    plotportfolio(1, portfolio2, len(srsi), 'plotportfolio1gb'+str(i), x)
    plotportfolio(0, portfolio2, len(srsi), 'plotportfolio0gb'+str(i), x)
    expava = ExpAv(0.0125, x[0], x)
    plotexpav(t, expava, 'plotexpavb'+str(i))
    expavag = ExpAv(0.0125, xgbm[1], xgbm)
    plotexpav(t, expavag, 'plotexpavgb'+str(i))
for i in range(0, 1):
    run(x, N, dt, delta, dxL, delta2, dxH, t, i, portfolio, 1)
