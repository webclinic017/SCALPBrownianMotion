import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from math import sqrt
from math import floor
from matplotlib.finance import candlestick_ohlc
import csv

plt.rc('text', usetex=True)

#Parameters brownian()
#
#x[0]:      X(0), initial stock price
#N:         number of increments
#T:         time period
#dt:        time step
#delta:     "speed" of the Brownian motion, variance = delta**2t
N = 900
T = 150
dt = T/N
delta = .01
delta2 = .05
x = np.empty((N+1))
xL = np.empty((N+1))
xH = np.empty((N+1))
dxL = np.empty((N+1))
dxH = np.empty((N+1))
buya = np.empty((N+1))
x[0] = 10
xL[0] = 0.0
xH[0] = xL[0]
t = np.arange(0, T, dt)


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
sigma = 0.2

def GBM(x0, mu, sigma, x, T, N):
    t = np.linspace(0., 1., N+1)
    S = []
    S.append(x0)
    for i in range(1, N+1):
        dr = (mu - 0.5 * sigma**2)*t[i]
        diff = sigma * x[i-1]
        S.append(x0*np.exp(dr+diff))
    return S, t

def createportfolio(LEN, srsit, qty, qty2):
    portfolio = np.zeros(shape=(LEN,3))
    portfolio[srsit,1] = 0
    for i in range(0, srsit+1) :
        portfolio[i,0] = qty #initial condition (qty of money 0)
        portfolio[i,1] = qty2 #initial condition (qty of money 1)
    return portfolio

#Parameters stochrsi()
#
#x:         numpy array, prices (brownian, GBM, real datas)
#i:         stochrsi() computes the stochrsi of x at i
#stochrsit: stochrsi based on the last stochrsit prices, condition : stochrsit=n*dt, where n is a natural number and stochrsit is a natural number
stochrsit = 20
srsifiltered = np.empty((N-stochrsit))
portfolio = createportfolio(N+1, stochrsit, 1, 0)

def stochrsi(x, i, stochrsit):
    xH = np.amax(x[i-stochrsit:i])
    xL = np.amin(x[i-stochrsit:i])
    return (x[i-1]-xL)/(xH-xL)

#Paramters stochrsib()
#
#srsi:  numpy array of the values calculated by stochrsi()
#alpha: treshold for the buying point
#beta:  treshold for the selling point, |0.5-c|=alpha, -|0.5-c|=beta
alpha = 0.4
beta = 0.6

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

def plotportfoliog(i, portfolio, lsrsi, name, x, stochrsit):
    fig, ax = plt.subplots()
    tplub = np.arange(stochrsit, lsrsi+stochrsit)
    plt.xlabel(r'\textbf{time} ($k = \frac{t}{\Delta t}$)')
    if i==0 or i==1:
        plt.title(r'EQPortfolio(t,'+str(1-i)+')')
    elif i>=2:
        plt.title(r'Portfolio(t, '+str(i-2)+')')
    p = np.delete([tuple([portfoliogain(i, j, portfolio, x)]) for j in range(((portfolio.shape[0])))], 1, axis=0)
    plt.plot(range(portfolio.shape[0]-1), p)
    plt.savefig(name+'.png')

def portfoliogain(i, j, portfolio, x):
    if i == 1:
        return portfolio[j, 0] + portfolio[j, 2]*portfolio[j, 1]
    elif i == 0:
        return portfolio[j, 0]/portfolio[j, 2] + portfolio[j, 1]
    elif i >=2:
        return portfolio[j, i-2]

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
    elif b == 0:
        plt.title(r'$B(t, \delta)$')
    else:
        plt.title(r'STOCK PRICE')
    plt.savefig(name + '.png')

def stochrsiarray(x, t, stochrsit):
    return np.delete([tuple([stochrsi(x, i, stochrsit)]) for i in range(stochrsit,len(t)+1)], 1, axis=0)

#Parameters method1()
#
#C:         treshold to sell and continue the process, C should depend on indicators at each time (tip : if you don't want to use it, make C large)
def c1(portfolio, srsib, expava, sl, cd, i, b, stochrsit, x):
    if b==0:
        return portfolio[i, 0] > 0 and srsib[i-stochrsit] == 0 and expava[i]<expava[i-1]
    elif b==1:
        return portfolio[i, 1] > 0 and x[i]>cd*buya[i-1] and srsib[i-stochrsit] == 1 and (x[i]*portfolio[i-1,1]+portfolio[i-1,0])<sl*portfoliogain(1, i-1, portfolio, x)

def method1(stochrsit, t, x, portfolio, srsib, srsi, dt, C, expava, arava, name, sl, cd):
    for i in range(stochrsit, len(srsi)+stochrsit):
        portfolio[i, 2] = x[i]
        if portfoliogain(1, i, portfolio, x) < C*portfolio[0, 0]:
            if c1(portfolio, srsib, expava, sl, cd, i, 0, stochrsit, x):
                if(buy(x, i+1, (1-srsib[i-stochrsit])*portfolio[i,0])[1] > [0.0]):
                    print("B " + str(buy(x, i+1, (1-srsib[i-stochrsit])*portfolio[i,0])[1]) + " F " + str(buy(x, i+1, (1-srsib[i-stochrsit])*portfolio[i,0])[0]) + " AT " + str(x[i+1]))
                portfolio[i+1,0] = portfolio[i, 0] + buy(x, i+1, (1-srsib[i-stochrsit])*portfolio[i,0])[0]
                portfolio[i+1,1] = portfolio[i, 1] + buy(x, i+1, (1-srsib[i-stochrsit])*portfolio[i,0])[1]
                buya[i] = x[i]
                with open(name+".txt", "a") as myfile:
                    myfile.write("B("+str(i)+") : " + str(portfolio[i,0]) + ", " + str(portfolio[i,1]) + " -> " + str(portfolio[i+1,0])+", "+str(portfolio[i+1,1]) + "*" + str(x[i+1]) + "\n")
            elif c1(portfolio, srsib, expava, sl, cd, i, 1, stochrsit, x):
                if(sell(x, i+1, (srsib[i-stochrsit])*portfolio[i,1])[0] > [0.0]):
                    print("S " + str(sell(x, i+1, (srsib[i-stochrsit])*portfolio[i,1])[1]) + " F " + str(sell(x, i+1, (srsib[i-stochrsit])*portfolio[i,1])[0]) + " AT " + str(x[i+1]) + " V " + str(portfoliogain(1, i, portfolio, x)))
                portfolio[i+1,0] = portfolio[i,0] + sell(x, i+1, srsib[i-stochrsit]*portfolio[i, 1])[0]
                portfolio[i+1,1] = portfolio[i,1] + sell(x, i+1, srsib[i-stochrsit]*portfolio[i, 1])[1]
                with open(name+".txt", "a") as myfile:
                    myfile.write("S("+str(i)+") : " + str(portfolio[i, 0]) + ", " + str(portfolio[i, 1]) + " -> " + str(portfolio[i+1,0]) + ", " + str(portfolio[i+1,1]) + "*" + str(x[i+1])+ "\n")
                buya[i] = buya[i-1]
            else:
                portfolio[i+1, 0] = portfolio[i, 0]
                portfolio[i+1, 1] = portfolio[i, 1]
                buya[i] = buya[i-1]
                with open(name+".txt", "a") as myfile:
                    myfile.write("K("+str(i)+") : " + str(portfolio[i, 0]) + ", " + str(portfolio[i, 1])+ " -> " + str(portfolio[i+1,0]) + ", " + str(portfolio[i+1,1])+"*"+str(x[i+1])+"\n")
        else:
            portfolio[i+1, 0] = portfolio[i, 0] + sell(x, i+1, portfolio[i, 1])[0]
            portfolio[i+1, 1] = portfolio[i, 1] + sell(x, i+1, portfolio[i, 1])[1]
            portfolio[len(srsi)+stochrsit, 0] = portfolio[i + 1, 0]
            portfolio[len(srsi)+stochrsit, 1] = portfolio[i + 1, 1]
            C += 0.0
            i = len(srsi)+stochrsit
        portfolio[len(srsi)+stochrsit,2]=x[len(srsi)+stochrsit]
    return portfolio

def game(x, N, dt, delta, t, portfolio, dxH, dxL, j):
    brownian(x[0], N, dt, delta, out=x[1:])
    xgbm = GBM(x[0], mu, sigma, x, T, N)[0]
    plotbrownian(xgbm, dxH*0, dxL*0, dt, t, 'GBMGAME'+str(j), 1)
    portfolio[len(xgbm)-1, 2] = xgbm[len(xgbm)-1]
    for i in range(1, len(xgbm)-1):
        portfolio[i, 2] = xgbm[i]
        print(portfoliogain(1, i, portfolio, xgbm))
        print("P " + str(xgbm[i]))
        command = input("?")
        portfolio[i, 2] = xgbm[i]
        if(command=="b"):
            portfolio[i+1,0] = portfolio[i, 0] + buy(xgbm, i+1, portfolio[i,0])[0]
            portfolio[i+1,1] = portfolio[i, 1] + buy(xgbm, i+1, portfolio[i,0])[1]
        elif(command=="s"):
            portfolio[i+1,0] = portfolio[i,0] + sell(xgbm, i+1, (portfolio[i, 1]))[0]
            portfolio[i+1,1] = portfolio[i,1] + sell(xgbm, i+1, (portfolio[i, 1]))[1]
        else:
            portfolio[i+1, 0] = portfolio[i, 0]
            portfolio[i+1, 1] = portfolio[i, 1]

    plotportfoliog(1, portfolio, len(portfolio), 'PFGAME'+str(j), x)

def get_data_csv(filename):
    prices = []
    with open(filename, 'r') as csvf:
        csvFR = csv.reader(csvf)
        next(csvFR)
        for row in csvFR:
            prices.append(float(row[1]))
    return prices

def BestRatio(x):
    s = 1
    p = 0
    for k in range(1, len(x)-1):
        if(x[k]>x[k-1] and x[k]>x[k+1]):
            s = s*x[k]
            p += 1
        elif(x[k]<x[i-1] and x[k]<x[i+1]):
            s = s/x[k]
            p += 1
    if p/2 != floor(p/2):
        s = s/x[len(x)-1]
    return s;

def runBrownian(x, N, dt, delta, dxL, delta2, dxH, t, i, portfolio, alpha, beta, C, sl, cd):
    k = 0
    brownian(x[0], N, dt, delta, out=x[1:])
    '''brownian(dxL[0], N, dt, delta2, out=dxL[1:])
    brownian(dxH[0], N, dt, delta2, out=dxH[1:])'''
    '''plotbrownian(x, dxH*0, dxL*0, dt, t, 'B'+str(i), 0)'''
    xgbm = GBM(x[0], mu, sigma, x, T, N)[0]
    plotbrownian(xgbm, dxH*0, dxL*0, dt, t, 'GBM'+str(i), 1)
    srsi = stochrsiarray(x, t, stochrsit)
    srsib = stochrsib(srsi, alpha, beta)
    srsig = stochrsiarray(xgbm, t, stochrsit)
    srsigb = stochrsib(srsi, alpha, beta)
    expava = EMA(0.1, x[0], x)
    expavag = EMA(0.1, xgbm[1], xgbm)
    """plotkav(t, expavag, 'EXPAVGB'+str(i), 0)
    plotkav(t, expava, 'EXPAVB'+str(i), 0)"""
    """arava = SMA(x, 10, dt)"""
    """plotkav(arava[0], arava[1], 'ARAVAB'+str(i), 1)"""
    aravag = SMA(xgbm, 10, dt)
    """plotkav(aravag[0], aravag[1], 'ARAVAGB'+str(i), 1)
    plotsrsifiltered(t, srsigb, stochrsit, 'SRSIGF'+str(i))
    plotsrsi(t, srsig, stochrsit, 'plotsrsig'+str(i))
    plotsrsifiltered(t, srsib, stochrsit, 'SRSIF'+str(i))
    plotsrsi(t, srsi, stochrsit, 'plotsrsi'+str(i))"""
    """print("RUNBM"+str(i))"""
    '''portfolio = method1(stochrsit, t, x, portfolio, srsib, srsi, dt, 1.1, expava, 0, "run"+str(i), sl, cd)'''
    """if(portfoliogain(1, len(srsi)+stochrsit, portfolio, x)>portfolio[stochrsit,0]):
        k += 1"""
    '''plotportfoliog(1, portfolio, len(srsi), 'PF0BMEQ'+str(i), x, stochrsit)'''
    """plotportfoliog(0, portfolio, len(srsi), 'PF1BMEQ'+str(i), x, stochrsit)"""
    print("RUNGBM"+str(i))
    portfolio2 = method1(stochrsit, t, xgbm, portfolio, srsigb, srsig, dt, 1.1, expavag, 0, "rung"+str(i), sl, cd)
    """if(portfoliogain(1, len(srsi)+stochrsit, portfolio2, x)>portfolio2[stochrsit,0]):
        k += 1"""
    """plotportfoliog(0, portfolio2, len(srsi), 'PF1GBMEQ'+str(i), x, stochrsit)"""
    plotportfoliog(1, portfolio2, len(srsi), 'PF0GBMEQ'+str(i), x, stochrsit)
    """plotportfoliog(2, portfolio, len(srsi), 'PF0BM'+str(i), x, stochrsit)
    plotportfoliog(2, portfolio2, len(srsi), 'PF0GBM'+str(i), x, stochrsit)"""

def runBacktest(srsit, C, sl, cd, i, data, qty, qty2):
    pdata = get_data_csv(data)
    portfolioBt = createportfolio(len(pdata), srsit, qty, qty2)
    t = np.arange(0, len(pdata)-1, 1)
    srsip = stochrsiarray(pdata, t , srsit)
    srsibp = stochrsib(srsip, alpha, beta)
    emap = EMA(0.1, pdata[1], pdata)
    portfolioBt = method1(srsit,  t, pdata, portfolioBt, srsibp, srsip, 1, C, emap, 0, "run"+str(data)+str(i), sl, cd)
    plotportfoliog(1, portfolioBt, len(srsip), 'PF0'+str(data)+str(i), x, 2)
    plotbrownian(pdata, dxH*0, dxL*0, 1, t, str(data)+str(i), 2)

for i in range(0, 1):
    runBrownian(x, N, dt, delta, dxL, delta2, dxH, t, i, portfolio, alpha, beta, 1.1, 1.01, 0.95)
    """game(x, N, dt, delta, t, portfolio, dxH, dxL, i)"""

runBacktest(2, 1.01, 1.005, 0.95, i, "GOOG.csv", 1, 0)
runBacktest(2, 1.01, 1.005, 0.95, i, "AAPL.csv", 1, 0)
runBacktest(10, 1.32, 1.05, 0.99, i, "FB.csv", 1, 0)
