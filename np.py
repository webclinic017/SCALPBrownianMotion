import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from math import sqrt
from math import floor
from matplotlib.finance import candlestick_ohlc
import datetime

N = 400
USDBTC = np.zeros(shape=(N,4))
delta = 8
T = 1600
dt = T/N
x = np.empty((N+1))
xL = np.empty((N+1))
xH = np.empty((N+1))
dxL = np.empty((N+1))
dxH = np.empty((N+1))
t = np.arange(0, T, dt)
x[0] = 1000
xL[0] = 0.0
xH[0] = xL[0]

volatility = 16
stochrsit = 5
USDBTC[stochrsit,1] = 0
USDBTC[stochrsit,0] = 100
srsibin = np.empty((N-stochrsit))

def brownien(x0, n, dt, delta, out=None):
    x0 = np.asarray(x0)
    r = norm.rvs(size=x0.shape + (n,), scale=delta*sqrt(dt))
    if out is None:
        out = np.empty(r.shape)
    np.cumsum(r, axis=-1, out=out)
    out += np.expand_dims(x0, axis=-1)
    return out

def stochrsi(x, i, stochrsit):
    xH = np.amax(x[i-stochrsit:i])
    xL = np.amin(x[i-stochrsit:i])
    return (x[i-1]-xL)/(xH-xL)

def achat(x, t, q):
    return [-q],[q/x[t]]

def vente(x, t, q):
    return [q*x[t]],[-q]

def plotsrsi(t, srsi, stochrsit, name):
    fig, ax = plt.subplots()
    srsi = [tuple([stochrsi(x, i, stochrsit)]) for i in range(stochrsit,len(t)+1)]
    srsi = np.delete(srsi, 1, axis=0)
    plt.plot(np.arange(stochrsit*dt, T, dt), srsi)
    plt.savefig(name+'.png')

def plotusdbtc(i,usdbtc, lsrsi, name):
    fig, ax = plt.subplots()
    tplub = np.arange(stochrsit, lsrsi+stochrsit-1)
    if i==1:
        plt.plot( np.arange(stochrsit, stochrsit+len(np.delete(usdbtc,np.s_[0:3], axis=1))*dt,dt), np.delete(usdbtc,np.s_[0:3], axis=1))
    else:
        plt.plot(np.arange(stochrsit, stochrsit+len(np.delete(usdbtc,np.s_[0:3], axis=1))*dt,dt), np.delete(np.delete(usdbtc,np.s_[0:2], axis=1), np.s_[3:4], axis=1))
    plt.savefig(name+'.png')

def plotsrsibin(t, srsibin, stochrsit, name):
    fig, ax = plt.subplots()
    plt.plot(np.arange(stochrsit*dt, T, dt), srsibin)
    plt.savefig(name+'.png')

def plotbrownien(x, dxH, dxL, dt, t, name):
    dxL = -abs(dxL)
    dxH = abs(dxH)
    for i in range(len(t)+1):
        dxL[i] *= (x[i]-x[i-1])/x[i]
        dxH[i] *= (x[i]-x[i-1])/x[i]
    quotes = [tuple([t[i],
                     x[i],
                     x[i+1]+dxH[i+1],
                     x[i+1]+dxL[i+1],
                     x[i+1]]) for i in range(len(t))]
    fig, ax = plt.subplots()
    candlestick_ohlc(ax, quotes, width=dt, colorup='g', colordown='r', alpha=1.0)
    plt.savefig(name + '.png')

def stochrsib(srsi):
    for i in range(0, len(srsi)):
        if(srsi[i]<0.5):
            srsibin[i] = 0
        elif(srsi[i]>0.5):
            srsibin[i] = 1
        else:
            srsibin[i]=srsi[i]
    return srsibin

def stochrsiarray(x, t, stochrsit):
    return np.delete([tuple([stochrsi(x, i, stochrsit)]) for i in range(stochrsit,len(t)+1)], 1, axis=0)

"""le choix de treshold dépendra des risques de l'action : risque qu'on souhaite encourir (à voir si je définis ça par rapport à un gain min ou pas seulement) (si élevé alors en vente : treshold très important, on hold plus, en vente : treshold élevé, on hold plus, &inversement), à réfléchir si on all-in à chaque fois ou si l'on met en jeu abs(1-srsi)*capital"""

def ag1(stochrsit, t, x, USDBTC, srsib, srsi, dt):
    for i in range(stochrsit, len(srsi)+stochrsit-1):
        if USDBTC[i, 0] != 0 and srsib[i-stochrsit] == 0 :
            USDBTC[i+1,0] = USDBTC[i, 0] + achat(x, i+1, pow((1-srsi[i-stochrsit]),1)*USDBTC[i,0])[0]
            USDBTC[i+1,1] = USDBTC[i, 1] + achat(x, i+1, pow((1-srsi[i-stochrsit]),1)*USDBTC[i,0])[1]
            with open("run.txt", "a") as myfile:
                myfile.write("USDBTC[][]B : " + str(USDBTC[i,0])+","+str(USDBTC[i,1])+ " -> "+ str(USDBTC[i+1,0])+","+str(USDBTC[i+1,1])+"*"+str(x[i+1]) + "\n")
        elif USDBTC[i, 1] != 0 and srsib[i-stochrsit] == 1:
            USDBTC[i+1,0] = USDBTC[i,0] + vente(x, i+1, pow(srsi[i-stochrsit]*USDBTC[i,1],1))[0]
            USDBTC[i+1,1] = USDBTC[i,1] + vente(x, i+1, pow(srsi[i-stochrsit]*USDBTC[i,1],1))[1]
            with open("run.txt", "a") as myfile:
                myfile.write("USDBTC[][]S : " + str(USDBTC[i,0])+","+str(USDBTC[i,1])+ " -> "+str(USDBTC[i+1,0])+","+str(USDBTC[i+1,1])+"*"+str(x[i+1])+ "\n")
        else :
            USDBTC[i+1,0] = USDBTC[i,0]
            USDBTC[i+1,1] = USDBTC[i,1]
            with open("run.txt", "a") as myfile:
                myfile.write("USDBTC[][]K : " + str(USDBTC[i,0])+","+str(USDBTC[i,1])+ " -> "+str(USDBTC[i+1,0])+","+str(USDBTC[i+1,1])+"*"+str(x[i+1])+"\n")
        USDBTC[i,2] = USDBTC[i,0]+x[i]*USDBTC[i,1]
        USDBTC[i,3] = USDBTC[i,0]/x[i]+USDBTC[i,1]
    return USDBTC


"""for i in range(stochrsit, len(t)-stochrsit-1):
    print(str(USDBTC[i,0])+'   '+str(USDBTC[i,1]))"""
def run(x, N, dt, delta, dxL, volatility, dxH, t, i, USDBTC):
    brownien(x[0], N, dt, delta, out=x[1:])
    brownien(dxL[0], N, dt, volatility, out=dxL[1:])
    brownien(dxH[0], N, dt, volatility, out=dxH[1:])
    plotbrownien(x, dxH*0, dxL*0, dt, t, 'plotb'+str(i))
    srsi = stochrsiarray(x, t, stochrsit)
    srsib = stochrsib(srsi)
    plotsrsibin(t, srsib, stochrsit, 'plotsrsibin'+str(i))
    plotsrsi(t, srsi, stochrsit, 'plotsrsi'+str(i))
    USDBTC = ag1(stochrsit, t, x, USDBTC, srsib, srsi, dt)
    plotusdbtc(1, USDBTC, len(srsi), 'plotusdbtc1'+str(i))
    plotusdbtc(0, USDBTC, len(srsi), 'plotusdbtc0'+str(i))
for i in range(0, 1):
    run(x, N, dt, delta, dxL, volatility, dxH, t, i, USDBTC)
