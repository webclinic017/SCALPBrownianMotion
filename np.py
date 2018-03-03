import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from math import sqrt
from math import floor
from matplotlib.finance import candlestick_ohlc
import datetime

plt.rc('text', usetex=True)

#Parameters brownian()
#
#x[0]:      X(0), initial stock price
#N:         number of increments
#T:         time period
#dt:        time step
#delta:     "speed" of the Brownian motion, variance = delta**2t
N = 600
T = 300
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
portfolio = np.zeros(shape=(N+1,3))

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
stochrsit = 120
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
#alpha: treshold for the buying point
#beta:  treshold for the selling point
alpha = 0.5
beta = 0.5

def stochrsib(srsi, alpha, beta):
    for i in range(0, len(srsi)):
        if(srsi[i]<alpha): #treshold
            srsifiltered[i] = 0
        elif(srsi[i]>=beta): #treshold
            srsifiltered[i] = 1
        else:
            srsifiltered[i]=srsi[i]
    return srsifiltered

#Parameters EMA()
#
#alpha:     smoothing constant
#x0:        initial condition
#x:         set of datas
expava = np.empty((N+1))

def EMA(alpha, x0, x):
    expava[0] = x0
    for i in range(1,len(x)):
        expava[i] = alpha*x[i] + (1-alpha)*expava[i-1]
    return np.delete(expava, 1, axis = 0)

#Parameters SMA()
#
#n          SMA() will calculate the arithmetic average on the last n values of x (for i in range(len(x)))
#x:         set of datas

def SMA(x, n, dt):
    arava = np.empty((N-n))
    t = np.arange(n*dt, T, dt)
    for j in range(1, len(arava)):
        arava[j] = arava[j-1] + (x[j]-x[j-n])/n
    return t, arava

def buy(x, t, q):
    return [-q],[q/x[t]]

def sell(x, t, q):
    return [q*x[t]],[-q]

def plotkav(t, k, name, i):
    fig, ax = plt.subplots()
    plt.plot(t, k)
    plt.xlabel(r'\textbf{time} ($t = k\Delta t$)')
    if i == 0:
        plt.title(r"EMA$(B(t)) =  \alpha B(t)+(1-\alpha) $EMA$({B(t-\Delta t)})$",
        fontsize=16, color='black')
    elif i == 1:
        plt.title(r"SMA$(B(t)) = $ SMA$(B(t-\Delta t))+\frac{B(t)-B(t-n)}{n}$",
        fontsize=16, color='black')
    elif i == 2:
        plt.title(r"SA$(B(t)) =  \frac{1}{t}\mathrm{SA}(B(t-\Delta t))*(t-\Delta)+B(t))$",
        fontsize=16, color='black')
    plt.savefig(name+'.png')

def plotsrsi(t, srsi, stochrsit, name):
    fig, ax = plt.subplots()
    plt.xlabel(r'\textbf{time} ($t = k\Delta t$)')
    plt.title(r"StochRSI$(t, \tau)$")
    srsi = np.delete([tuple([stochrsi(x, i, stochrsit)]) for i in range(stochrsit,len(t)+1)], 1, axis=0)
    plt.plot(np.arange(stochrsit*dt, T, dt), srsi)
    plt.savefig(name+'.png')

def plotportfoliog(i, portfolio, lsrsi, name, x):
    fig, ax = plt.subplots()
    tplub = np.arange(stochrsit, lsrsi+stochrsit)
    plt.xlabel(r'\textbf{time} ($k = \frac{t}{\Delta t}$)')
    if i==1:
        plt.title(r'EQPortfolio(t, 0)')
    elif i==0:
        plt.title(r'EQPortfolio(t, 1)')
    elif i==2:
        plt.title(r'Portfolio(t, 0)')
    elif i==3:
        plt.title(r'Portfolio(t, 1)')
    p = np.delete([tuple([portfoliogain(i, j, portfolio, x)]) for j in range(((portfolio.shape[0])))], 1, axis=0)
    print(p)
    plt.plot(range(portfolio.shape[0]-1), p)
    plt.savefig(name+'.png')

def portfoliogain(i, j, portfolio, x):
    if i == 1:
        return portfolio[j, 0] + portfolio[j, 2]*portfolio[j, 1]
    elif i == 0:
        return portfolio[j, 0]/portfolio[j, 2] + portfolio[j, 1]
    elif i == 2:
        return portfolio[j, 0]
    elif i == 3:
        return portfolio[j ,1]

def plotsrsifiltered(t, srsifiltered, stochrsit, name):
    fig, ax = plt.subplots()
    plt.plot(np.arange(stochrsit*dt, T, dt), srsifiltered)
    plt.xlabel(r'\textbf{time} ($t = k\Delta t$)')
    plt.title(r'FilteredStochRSI($\alpha$,$\beta$,$t$,$\tau$)')
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
    plt.xlabel(r'\textbf{time} ($t = k\Delta t$)')
    candlestick_ohlc(ax, quotes, width=dt, colorup='g', colordown='r', alpha=1.0)
    if b == 1:
        plt.title(r'$B_g(t, \sigma, \mu, B)$')
    else:
        plt.title(r'$B(t, \delta)$')
    plt.savefig(name + '.png')

def stochrsiarray(x, t, stochrsit):
    return np.delete([tuple([stochrsi(x, i, stochrsit)]) for i in range(stochrsit,len(t)+1)], 1, axis=0)

"""le choix de treshold dépendra des risques de l'action : risque qu'on souhaite encourir (à voir si je définis ça par rapport à un gain min ou pas seulement) (si élevé alors en vente : treshold très important, on hold plus, en vente : treshold élevé, on hold plus, &inversement), à réfléchir si on all-in à chaque fois ou si l'on met en jeu f(srsi)*capital"""
#Parameters method1()
#
#C:         treshold to sell and continue the process, C should depend on indicators at each time (tip : if you don't want to use it, make C = portfolio[0, 0]+1 for example)
C = 1.01

def method1(stochrsit, t, x, portfolio, srsib, srsi, dt, p, C, expava, arava):
    for i in range(stochrsit, len(srsi)+stochrsit):
        portfolio[i, 2] = x[i]
        if portfoliogain(1, i, portfolio, x) < C*portfolio[0, 0]:
            if portfolio[i, 0] > 0 and (srsib[i-stochrsit] == 0):
                portfolio[i+1,0] = portfolio[i, 0] + buy(x, i+1, (1-srsi[i-stochrsit])**p*portfolio[i,0])[0]
                portfolio[i+1,1] = portfolio[i, 1] + buy(x, i+1, (1-srsi[i-stochrsit])**p*portfolio[i,0])[1]
                with open("run.txt", "a") as myfile:
                    myfile.write("B("+str(i)+") : " + str(portfolio[i,0]) + ", " + str(portfolio[i,1]) + " -> " + str(portfolio[i+1,0])+", "+str(portfolio[i+1,1]) + "*" + str(x[i+1]) + "\n")
            elif portfolio[i, 1] > 0 and (srsib[i-stochrsit] == 1):
                portfolio[i+1,0] = portfolio[i,0] + sell(x, i+1, (srsi[i-stochrsit]*portfolio[i, 1])**p)[0]
                portfolio[i+1,1] = portfolio[i,1] + sell(x, i+1, (srsi[i-stochrsit]*portfolio[i, 1])**p)[1]
                with open("run.txt", "a") as myfile:
                    myfile.write("S("+str(i)+") : " + str(portfolio[i, 0]) + ", " + str(portfolio[i, 1]) + " -> " + str(portfolio[i+1,0]) + ", " + str(portfolio[i+1,1]) + "*" + str(x[i+1])+ "\n")
            else:
                portfolio[i+1, 0] = portfolio[i, 0]
                portfolio[i+1, 1] = portfolio[i, 1]
                with open("run.txt", "a") as myfile:
                    myfile.write("K("+str(i)+") : " + str(portfolio[i, 0]) + ", " + str(portfolio[i, 1])+ " -> " + str(portfolio[i+1,0]) + ", " + str(portfolio[i+1,1])+"*"+str(x[i+1])+"\n")
        else:
            portfolio[i+1, 0] = portfolio[i, 0] + sell(x, i+1, portfolio[i, 1])[0]
            portfolio[i+1, 1] = portfolio[i, 1] + sell(x, i+1, portfolio[i, 1])[1]
            i = len(srsi)+stochrsit
            C += 0.01
        portfolio[len(srsi)+stochrsit,2]=x[len(srsi)+stochrsit]
    return portfolio

def run(x, N, dt, delta, dxL, delta2, dxH, t, i, portfolio, p, alpha, beta, C):
    brownian(x[0], N, dt, delta, out=x[1:])
    '''brownian(dxL[0], N, dt, delta2, out=dxL[1:])
    brownian(dxH[0], N, dt, delta2, out=dxH[1:])'''
    plotbrownian(x, dxH*0, dxL*0, dt, t, 'plotb'+str(i), 0)
    xgbm = GBM(x[0], mu, sigma, x, T, N)[0]
    plotbrownian(xgbm, dxH*0, dxL*0, dt, t, 'plotgbm'+str(i), 1)
    srsi = stochrsiarray(x, t, stochrsit)
    srsib = stochrsib(srsi, alpha, beta)
    srsig = stochrsiarray(xgbm, t, stochrsit)
    srsigb = stochrsib(srsi, alpha, beta)
    expava = EMA(0.5, x[0], x)
    expavag = EMA(0.5, xgbm[1], xgbm)
    plotkav(t, expavag, 'plotexpavgb'+str(i), 0)
    plotkav(t, expava, 'plotexpavb'+str(i), 0)
    arava = SMA(x, 10, dt)
    plotkav(arava[0], arava[1], 'plotaravab'+str(i), 1)
    aravag = SMA(xgbm, 10, dt)
    plotkav(aravag[0], aravag[1], 'plotaravabg'+str(i), 1)
    plotsrsifiltered(t, srsigb, stochrsit, 'plotsrsigfiltered'+str(i))
    plotsrsi(t, srsig, stochrsit, 'plotsrsig'+str(i))
    plotsrsifiltered(t, srsib, stochrsit, 'plotsrsifiltered'+str(i))
    plotsrsi(t, srsi, stochrsit, 'plotsrsi'+str(i))
    portfolio = method1(stochrsit, t, x, portfolio, srsib, srsi, dt, p, C, expava, arava)
    """plotportfoliog(0, portfolio, len(srsi), 'plotportfolio1b'+str(i), x)"""
    plotportfoliog(1, portfolio, len(srsi), 'plotportfolio0b'+str(i), x)
    portfolio2 = method1(stochrsit, t, xgbm, portfolio, srsigb, srsig, dt, p, C, expavag, aravag)
    """plotportfoliog(0, portfolio2, len(srsi), 'plotportfolio1gb'+str(i), x)"""
    plotportfoliog(1, portfolio2, len(srsi), 'plotportfolio0gb'+str(i), x)
for i in range(0, 1):
    run(x, N, dt, delta, dxL, delta2, dxH, t, i, portfolio, 1, alpha, beta, C)
