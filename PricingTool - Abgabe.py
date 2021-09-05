import pathlib
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import datetime as dt
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tkinter import *


def feature_matrix_from_current_state(state, exercise, poly, N, d):
    """
    Berechne die ersten 4 LaGuerre (o.ä.) Polynome im aktuellen Zustand t_i
    und erhalte damit eine (n x 4) Merkmals Matrix der n simulierten Pfade
    -Eingang: n-dimensionaler Vektor des Zustandes zum Zeitpunkt t_i
    -Ausgang: Merkmals Matrix für die Longstaff Schartz Regression
    """
    features = tf.ones((N,1))
    
    for k in range(d):

        asset = state[k*N:(k+1)*N]
        if k == 0:
            mixed = asset
            maxx = asset
        else:
            mixed = tf.multiply(mixed, asset)
            maxx = tf.maximum(maxx, asset)
        
        if poly == 'Chebychev':                            # chebychev polynomials
            feature_1 = tf.pow(asset,1)    
            feature_2 = 2*tf.pow(asset,2)-1
            feature_3 = 4*tf.pow(asset,3)-3*feature_1
        elif poly == 'Hermite':                            # hermite polynomials
            feature_1 = 2*tf.pow(asset,1)    
            feature_2 = 4*tf.pow(asset,2)-2
            feature_3 = 8*tf.pow(asset,3)-12*feature_1
        elif poly == 'Legendre':                           # legendre polynomials
            feature_1 = tf.pow(asset,1)    
            feature_2 = 0.5*(3*tf.pow(asset,2)-1)
            feature_3 = 0.5*(5*tf.pow(asset,3)-3*feature_1)        
        elif poly == 'Laguerre':                           # laguerre polynomials
            feature_1 = tf.exp(-asset/2)   
            feature_2 = tf.exp(-asset/2)*(1 - asset) 
            feature_3 = tf.exp(-asset/2)*(1 - 2*asset + tf.pow(asset,2)/2)
        elif poly == 'mixed_exc':
            feature_1 = tf.pow(asset,1)  
            feature_2 = tf.pow(asset,2)                       
        else:
            feature_1 = tf.pow(asset,1)  
            feature_2 = tf.pow(asset,2) 
            feature_3 = tf.pow(asset,3)      

        feature_1 = tf.reshape(feature_1, shape=(N, 1))

        feature_2 = tf.reshape(feature_2, shape=(N, 1))
        
        if poly == 'mixed_exc':
            features = tf.concat([features, feature_1, feature_2], axis=1)
        else:
            feature_3 = tf.reshape(feature_3, shape=(N, 1))
            features = tf.concat([features, feature_1, feature_2, feature_3], axis=1)
        
        
    if poly in ('mixed','mixed_max','mixed_squared','mixed_squared_exc','mixed_exc'):
        mixed = tf.reshape(mixed, shape=(N, 1))
        features = tf.concat([features, mixed], axis=1)

    if poly in ('mixed_max',):
        maxx = tf.reshape(maxx, shape=(N, 1))
        features = tf.concat([features, maxx], axis=1)

    if poly in ('mixed_squared','mixed_squared_exc'):
        for h in range(d):
            asset = tf.reshape(state[h*N:(h+1)*N], shape=(N, 1), name='pre_reshape')
            squared = tf.multiply(mixed, asset)
            squared = tf.reshape(squared, shape=(N, 1), name='reshape_squared')
            features = tf.concat([features, squared], axis=1)

    if poly in ('mixed_squared_exc','mixed_exc'):
        exercise = tf.reshape(exercise, shape=(N, 1))
        features = tf.concat([features, exercise], axis=1)
        
    return features

def feature_matrix(state, exercise, poly, N, d):
    X_reg = np.ones((N,1))   
            
    if poly == 'Chebychev':                                             # erstelle Design-Matrix M --- main part
        basis = np.c_[state**1, 2*state**2-1, 4*state**3 - 3 * state]   
    elif poly == 'Hermite':
        basis = np.c_[2*state, 4*state**2-2, 8*state**3 - 12 * state]   # Setze asset werte in basis polynome ein
    elif poly == 'Legendre':
        basis = np.c_[state**1, 0.5*(3*state**2 - 1), 0.5*(5*state**3 - 3 * state)]
    elif poly == 'Laguerre':
        basis = np.c_[np.exp(-state/2), np.exp(-state/2)*(1-state), np.exp(-state/2)*(1 - 2*state + (state**2)/2)]
    elif poly == 'mixed_exc':
        basis = np.c_[state**1, state**2]
    else:               
        basis = np.c_[state**1, state**2, state**3]                     # standart basis functions x, x^2, x^3 
        

    for j in range(d):
        X_reg = np.c_[X_reg, basis[j*N:(j+1)*N,:]]
        
        asset = state[j*N:(j+1)*N]
        if j == 0:
            mixed = asset
            maxx = asset
        else:
            mixed = mixed * asset
            maxx = np.maximum(maxx, asset)
    
    if poly in ('mixed','mixed_max','mixed_squared','mixed_squared_exc','mixed_exc'):
        X_reg = np.c_[X_reg, mixed]
        
    if poly in ('mixed_max',):
        X_reg = np.c_[X_reg, maxx]
        
    if poly in ('mixed_squared','mixed_squared_exc'):
        for h in range(d):
            squared = mixed * state[h*N:(h+1)*N]
            X_reg = np.c_[X_reg, squared]
        
    if poly in ('mixed_squared_exc','mixed_exc'):
        X_reg = np.c_[X_reg, exercise]               

    return X_reg

###############################################################
### AB HIER CODE FÜR BEPREISUNG MIT NACH LONGSTAFF SCHWARTZ ###
###############################################################

def pricing_function(number_call_dates, poly, N_train, N_pricing, d):
    S = tf.placeholder(tf.float32)
    dts = tf.placeholder(tf.float32)
    K = tf.placeholder(tf.float32)
    r = tf.placeholder(tf.float32)
    sigma = tf.placeholder(tf.float32)
    delta = tf.placeholder(tf.float32)
    dW = tf.placeholder(tf.float32)
    train_weights = tf.placeholder(tf.float32)
    part = tf.placeholder(tf.int32)
    
    for j in range(d):
        vola = sigma[j]
        dW_j = dW[j*part:(j+1)*part,:]
        
        S_t_temp = S * tf.cumprod(tf.exp((r-delta-vola**2/2)*dts + vola*tf.sqrt(dts)*dW_j), axis=1)
        E_t_temp = tf.exp(-r*tf.cumsum(dts))*tf.maximum(S_t_temp-K, 0)
            
        if j == 0:
            S_t = S_t_temp
            E_t = E_t_temp
        else:
            S_t = tf.concat([S_t, S_t_temp], axis=0, name='concatS_t')
            E_t = tf.maximum(E_t, E_t_temp, name='Max2')      
    
    
    continuationValues = []
    exercises = []
    previous_exercises = 0
    npv = 0
    for i in range(number_call_dates-1):
        X = feature_matrix_from_current_state(S_t[:,i], E_t[:,i], poly, N_pricing, d)
        w = train_weights[-i]
        contValue = tf.matmul(X, w)
        continuationValues.append(contValue)
        inMoney = tf.cast(tf.greater(E_t[:,i], 0.), tf.float32)
        exercise = tf.cast(tf.greater(E_t[:,i], contValue[:,0]), tf.float32) * inMoney 
        exercises.append(exercise)
        exercise = exercise * (1-previous_exercises)
        previous_exercises += exercise
        npv += exercise*E_t[:,i]
    
    # Last exercise date
    inMoney = tf.cast(tf.greater(E_t[:,-1], 0.), tf.float32)          # Underlying-Werte > 0 z.Z. t=T -> 1, ansonsten 0
    exercise =  inMoney * (1-previous_exercises)                      # Wenn zuvor schon ausgeübt -> 0, ansonsten 1
    npv += exercise*E_t[:,-1]                                         # Cashflow in t=T * Ausübungen in t=T, addiere zum npv
    npv = tf.reduce_mean(npv)                                         # Mittelwert über alle Ausübungswerte/Cashflows
    greeks = tf.gradients(npv, [S, r, sigma])                         # Greeks = Ableitung von npv in einzelne Variablen
    return([S, dts, K, r, sigma, delta, dW, S_t, E_t, npv, greeks, train_weights, part])

def bermudanMC_tensorFlow(S_0, strike, T, calldates, dividend, impliedvol, riskfree_r, random_train, random_pricing, plots, poly):
    """
    Calculate the npv (net present value) of a bermudan max call option
    returns:    npv, (delta, rho, vega)
    """
    d = len(impliedvol)
    N_pricing = int(len(random_pricing)/d)
    N_train = int(len(random_train)/d)
    exTimes = []
    dist = T/calldates
    k = 1
    while k <= calldates:                                       # Erstelle Liste mit äquidistanten Ausübungszeitpunkten
        exTimes.append(dist)
        k = k + 1
    
    n_exercises = len(exTimes)
    train_functions = []
    
    with tf.Session() as sess:                                  # eröffne Berechnungsumgebung (session)
        # berechne Pfade und npv´s zu jeder Zeit t
        S, dts, K, r, sigma, delta, dW, S_t, E_t, npv, greeks, train_weights, part = pricing_function(n_exercises, poly, N_train, N_pricing, d)
        sess.run(tf.global_variables_initializer())             # initialisiere ops 
        paths, exercise_values = sess.run([S_t, E_t], {  
            S: S_0,                                             # verknüpfe alle inputs mit den Variablen in der Berechnung
            dts : exTimes,
            K : strike,
            r : riskfree_r,
            sigma: impliedvol,
            delta: dividend,
            dW : random_train,
            train_weights : train_functions,
            part : N_train
        })

        # Starte Rückwärtsiteration um Fortführungswerte-Regression für jeden Ausübungszeitpunkt zu erhalten
        cont_value = np.zeros((N_train,1)) 
        
        for i in range(n_exercises-1)[::-1]:                                # starte beim vorletzen Ausübungszeitpunkt 
            y = exercise_values[:, i+1:i+2]
            X = paths[:, i]                                                 # Underlying Werte zum Zeitpunkt i      
            X_reg = feature_matrix(X, exercise_values[:, i:i+1], poly, N_train, d)   
            V = np.maximum(y, cont_value)
            X_trans = np.transpose(X_reg[exercise_values[:,i]>0])           # nur in the money pfade benutzen
            w = np.matmul(np.matmul(np.linalg.inv(np.matmul(X_trans,X_reg[exercise_values[:,i]>0])),X_trans),V[exercise_values[:,i]>0]) 
            train_functions.append(w)
            cont_value = np.matmul(X_reg,w)
            exercise_values[:, i:i+1] = np.maximum(exercise_values[:, i:i+1], cont_value)  # Ausübungspreis berechnen

            
            if plots == 'Yes':     
                plt.figure(figsize=(10,10))
                plt.scatter(paths[:,i], y, color='blue')
                plt.scatter(paths[:,i], cont_value, color='red')
                plt.title('Continuation Value approx')
                plt.ylabel('NPV t_%i'%i)
                plt.xlabel('S_%i'%i)
            
        # Forward simulation mit trainierten Gewichten w
        npv, greeks = sess.run([npv, greeks], { S: S_0,
                                                dts : exTimes,
                                                K : strike,
                                                r : riskfree_r,
                                                sigma: impliedvol,
                                                delta: dividend,
                                                dW : random_pricing,
                                                train_weights : train_functions,
                                                part : N_pricing
                                              })
        return(npv, greeks)


#######################################################
### AB HIER CODE FÜR BEPREISUNG MIT OBERER SCHRANKE ###
#######################################################

def pricing_function_wUB(number_call_dates, poly, N_train, N_pricing, d):
    S = tf.placeholder(tf.float32)
    dts = tf.placeholder(tf.float32)
    K = tf.placeholder(tf.float32)
    r = tf.placeholder(tf.float32)
    sigma = tf.placeholder(tf.float32)
    delta = tf.placeholder(tf.float32)
    dW = tf.placeholder(tf.float32)
    train_weights = tf.placeholder(tf.float32)
    part = tf.placeholder(tf.int32)
    
    for j in range(d):
        vola = sigma[j]
        dW_j = dW[j*part:(j+1)*part,:]
        
        S_t_temp = S * tf.cumprod(tf.exp((r-delta-vola**2/2)*dts + vola*tf.sqrt(dts)*dW_j), axis=1)
        E_t_temp = tf.exp(-r*tf.cumsum(dts))*tf.maximum(S_t_temp-K, 0)
            
        if j == 0:
            S_t = S_t_temp
            E_t = E_t_temp
        else:
            S_t = tf.concat([S_t, S_t_temp], axis=0, name='concatS_t')
            E_t = tf.maximum(E_t, E_t_temp, name='Max2')      
     
    continuationValues = []
    exercises = []   
    previous_exercises = 0
    npv = 0
    for i in range(number_call_dates-1):
        X = feature_matrix_from_current_state(S_t[:,i], E_t[:,i], poly, N_pricing, d)
        w = train_weights[-i]
        contValue = tf.matmul(X, w)
        continuationValues.append(contValue)
        inMoney = tf.cast(tf.greater(E_t[:,i], 0.), tf.float32)
        exercise = tf.cast(tf.greater(E_t[:,i], contValue[:,0]), tf.float32) * inMoney
        
        if i > 0:  # nested part  
            V_max = tf.maximum( tf.maximum(E_t[:,i],0) , contValue[:,0])            
            
            N_nested = 10
            randoms = np.random.randn(N_pricing*d, N_nested)

            for j in range(d):     # new paths / successors
                vola = sigma[j]
                randoms_j = randoms[j*N_pricing:(j+1)*N_pricing,:]
                
                S_t_temp = tf.transpose(tf.exp((r-delta-vola**2/2)*dts[0] + vola*tf.sqrt(dts[0])*randoms_j)) * S_t[j*N_pricing:(j+1)*N_pricing, i-1]
                E_t_temp = tf.exp(-r*dts[0])*tf.maximum( tf.transpose(S_t_temp) - K, 0)

                if j == 0:
                    S_t_nested = tf.transpose(S_t_temp)
                    E_t_nested = E_t_temp
                else:
                    S_t_nested = tf.concat([S_t_nested, tf.transpose(S_t_temp)], axis=0, name='concatS_t_nest')
                    E_t_nested = tf.maximum(E_t_nested, E_t_temp, name='Max_nest')    

            for k in range(N_nested):         # calculate estimator for V   
                X_nested = feature_matrix_from_current_state(S_t_nested[:,i], E_t_nested[:,i], poly, N_pricing, d)
                contValue_nested = tf.matmul(X_nested, w, name='matmul_nested_1')
                V_temp = tf.maximum( tf.maximum(E_t_nested[:,i], 0.) , contValue_nested[:,0]) 
                                            
                if k == 0:
                    V_nested = V_temp
                else:
                    V_nested = tf.add(V_nested, V_temp)
                    
            Diff = V_max - V_nested / N_nested 
            
            if i == 1:
                Martingal = Diff
                V_0 = tf.add(E_t[:,i], (-1)*Martingal)
            else:
                Martingal += Diff
                V_0 = tf.maximum(V_0, tf.add(E_t[:,i], (-1)*Martingal) )
       
        exercises.append(exercise) 
        exercise = exercise * (1-previous_exercises)
        previous_exercises += exercise
        npv += exercise*E_t[:,i]
    
    # Last exercise date
    inMoney = tf.cast(tf.greater(E_t[:,-1], 0.), tf.float32) # Underlying-Werte > 0 z.Z. t=T -> 1, ansonsten 0
    V_max = tf.maximum(E_t[:,-1], 0.)               # last step nested
    randoms = np.random.randn(N_pricing*d, N_nested)
    
    for j in range(d):
        vola = sigma[j]
        randoms_j = randoms[j*N_pricing:(j+1)*N_pricing,:]
                
        S_t_temp = tf.transpose(tf.exp((r-delta-vola**2/2)*dts[0] + vola*tf.sqrt(dts[0])*randoms_j)) * S_t[j*N_pricing:(j+1)*N_pricing, i-1]
        E_t_temp = tf.exp(-r*dts[0])*tf.maximum( tf.transpose(S_t_temp) - K, 0)
                
        if j == 0:
            S_t_nested = tf.transpose(S_t_temp)
            E_t_nested = E_t_temp
        else:
            S_t_nested = tf.concat([S_t_nested, tf.transpose(S_t_temp)], axis=0, name='concatS_t_nest_2')
            E_t_nested = tf.maximum(E_t_nested, E_t_temp, name='Max_nest_2')
    
    for k in range(N_nested):
        X_nested = feature_matrix_from_current_state(S_t_nested[:,i], E_t_nested[:,i], poly, N_pricing, d)
        contValue_nested = tf.matmul(X_nested, w, name='Matmul_nested')
        V_temp = tf.maximum( tf.maximum(E_t_nested[:,i], 0.) , contValue_nested[:,0]) 
                                            
        if k == 0:
            V_nested = V_temp
        else:
            V_nested = tf.add(V_nested, V_temp)
                    
    Diff = V_max - V_nested / N_nested    
    Martingal += Diff    
    V_0 = tf.maximum(V_0, tf.add(E_t[:,i], (-1)*Martingal) )       
    UB = tf.reduce_mean(V_0)
    
    exercise =  inMoney * (1-previous_exercises)                      # Wenn zuvor schon ausgeübt -> 0, ansonsten 1
    npv += exercise*E_t[:,-1]                                         # Cashflow in t=T * Ausübungen in t=T, addiere zum npv
    npv = tf.reduce_mean(npv)                                         # Mittelwert über alle Ausübungswerte/Cashflows
    greeks = tf.gradients(npv, [S, r, sigma])                         # Greeks = Ableitung von npv in einzelne Variablen
    return([S, dts, K, r, sigma, delta, dW, S_t, E_t, npv, greeks, UB, train_weights, part])


def bermudanMC_tensorFlow_wUB(S_0, strike, T, calldates, dividend, impliedvol, riskfree_r, random_train, random_pricing, plots, poly):
    d = len(impliedvol)
    N_pricing = int(len(random_pricing)/d)
    N_train = int(len(random_train)/d)
    exTimes = []
    dist = T/calldates
    k = 1
    while k <= calldates:                                       # Erstelle Liste mit äquidistanten Ausübungszeitpunkten
        exTimes.append(dist)
        k = k + 1
    
    n_exercises = len(exTimes)
    train_functions = []
    
    with tf.Session() as sess:                                  # eröffne Berechnungsumgebung (session)
        # berechne Pfade und npv´s zu jeder Zeit t
        S, dts, K, r, sigma, delta, dW, S_t, E_t, npv, greeks, UB, train_weights, part = pricing_function_wUB(n_exercises, poly, N_train, N_pricing, d)
        sess.run(tf.global_variables_initializer())             # initialisiere ops 
        paths, exercise_values = sess.run([S_t, E_t], {  
            S: S_0,                                             # verknüpfe alle inputs mit den Variablen in der Berechnung
            dts : exTimes,
            K : strike,
            r : riskfree_r,
            sigma: impliedvol,
            delta: dividend,
            dW : random_train,
            train_weights : train_functions,
            part : N_train
        })

        # Starte Rückwärtsiteration um Fortführungswerte-Regression für jeden Ausübungszeitpunkt zu erhalten
        cont_value = np.zeros((N_train,1)) 
        
        for i in range(n_exercises-1)[::-1]:                                # starte beim vorletzen Ausübungszeitpunkt 
            y = exercise_values[:, i+1:i+2]
            X = paths[:, i]                                                 # Underlying Werte zum Zeitpunkt i
            
            X_reg = feature_matrix(X, exercise_values[:, i:i+1], poly, N_train, d)
            
            V = np.maximum(y, cont_value)
            X_trans = np.transpose(X_reg[exercise_values[:,i]>0])                               # nur in the money pfade benutzen
            w = np.matmul(np.matmul(np.linalg.inv(np.matmul(X_trans,X_reg[exercise_values[:,i]>0])),X_trans),V[exercise_values[:,i]>0]) 
            train_functions.append(w)
            cont_value = np.matmul(X_reg,w)
            exercise_values[:, i:i+1] = np.maximum(exercise_values[:, i:i+1], cont_value)   # Ausübungspreis berechnen
            
            print('train step' , i , ' done!')
            
            if plots == 'Yes':     
                plt.figure(figsize=(10,10))
                plt.scatter(paths[:,i], y, color='blue')
                plt.scatter(paths[:,i], cont_value, color='red')
                plt.title('Continuation Value approx')
                plt.ylabel('NPV t_%i'%i)
                plt.xlabel('S_%i'%i)
                
        
        # Forward simulation mit trainierten Gewichten w
        npv, UB = sess.run([npv, UB],         { S: S_0,
                                                dts : exTimes,
                                                K : strike,
                                                r : riskfree_r,
                                                sigma: impliedvol,
                                                delta: dividend,
                                                dW : random_pricing,
                                                train_weights : train_functions,
                                                part : N_pricing
                                              })
        return(npv, UB) 
    

#########################################################
### AB HIER CODE FÜR BEPREISUNG MIT NEURONALEN NETZEN ###
#########################################################

def pricing_function_NN(number_call_dates, poly, N_train, N_pricing, d):
    S = tf.placeholder(tf.float32)
    dts = tf.placeholder(tf.float32)
    K = tf.placeholder(tf.float32)
    r = tf.placeholder(tf.float32)
    sigma = tf.placeholder(tf.float32)
    delta = tf.placeholder(tf.float32)
    dW = tf.placeholder(tf.float32)
    train_weights = tf.placeholder(tf.string)
    part = tf.placeholder(tf.int32)
    
    for j in range(d):
        vola = sigma[j]
        dW_j = dW[j*part:(j+1)*part,:]
        
        S_t_temp = S * tf.cumprod(tf.exp((r-delta-vola**2/2)*dts + vola*tf.sqrt(dts)*dW_j), axis=1)
        E_t_temp = tf.exp(-r*tf.cumsum(dts))*tf.maximum(S_t_temp-K, 0)
            
        if j == 0:
            S_t = S_t_temp
            E_t = E_t_temp
        else:
            S_t = tf.concat([S_t, S_t_temp], axis=0, name='concatS_t')
            E_t = tf.maximum(E_t, E_t_temp, name='Max2')      
    
    continuationValues = []
    exercises = []
    previous_exercises = 0
    npv = 0
    for i in range(number_call_dates-1):
        X = feature_matrix_from_current_state(S_t[:,i], E_t[:,i], poly, N_pricing, d)
        model = build_model(X)
        checkpoint_path = train_weights[-i]
        try:
            model.load_weights(checkpoint_path)
            contValue = model.predict(X)
            continuationValues.append(contValue)
            inMoney = tf.cast(tf.greater(E_t[:,i], 0.), tf.float32)
            exercise = tf.cast(tf.greater(E_t[:,i], contValue[:,0]), tf.float32) * inMoney 
            exercises.append(exercise)
            exercise = exercise * (1-previous_exercises)
            previous_exercises += exercise
            npv += exercise*E_t[:,i]
        except AttributeError:
            pass
    
    # Last exercise date
    inMoney = tf.cast(tf.greater(E_t[:,-1], 0.), tf.float32)          # Underlying-Werte > 0 z.Z. t=T -> 1, ansonsten 0
    exercise =  inMoney * (1-previous_exercises)                      # Wenn zuvor schon ausgeübt -> 0, ansonsten 1
    npv += exercise*E_t[:,-1]                                         # Cashflow in t=T * Ausübungen in t=T, addiere zum npv
    npv = tf.reduce_mean(npv)                                         # Mittelwert über alle Ausübungswerte/Cashflows
    greeks = tf.gradients(npv, [S, r, sigma])                         # Greeks = Ableitung von npv in einzelne Variablen
    return([S, dts, K, r, sigma, delta, dW, S_t, E_t, npv, greeks, train_weights, part])

# build neural network with 64 nodes in Hidden Layer
def build_model(features):
    try: 
        model = keras.Sequential([layers.Dense(64, activation=tf.nn.relu, input_shape=[len(np.transpose(features))]),
                                  layers.Dense(64, activation=tf.nn.relu),
                                  layers.Dense(1)])                                     

        model.compile(loss = 'mean_squared_error',
                      optimizer = tf.keras.optimizers.Adam(0.01),
                      metrics = ['mean_absolute_error', 'mean_squared_error'])
    except TypeError:
        model = 0
    return model

# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end = '')
    

def bermudanMC_tensorFlow_NN(S_0, strike, T, calldates, dividend, impliedvol, riskfree_r, random_train, random_pricing, plots, poly):
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)      # if no improvement happend
  
    d = len(impliedvol)
    N_pricing = int(len(random_pricing)/d)
    N_train = int(len(random_train)/d)
    exTimes = []
    dist = T/calldates
    k = 1
    while k <= calldates:                                       # Erstelle Liste mit äquidistanten Ausübungszeitpunkten
        exTimes.append(dist)
        k = k + 1
    
    n_exercises = len(exTimes)
    train_functions = []
    train_stats = []
    
    with tf.Session() as sess:                                  # eröffne Berechnungsumgebung (session)
        # berechne Pfade und npv´s zu jeder Zeit t
        S, dts, K, r, sigma, delta, dW, S_t, E_t, npv, greeks, train_weights, part = pricing_function_NN(n_exercises, poly, N_train, N_pricing, d)
        sess.run(tf.global_variables_initializer())             # initialisiere ops 
        paths, exercise_values = sess.run([S_t, E_t], {  
            S: S_0,                                             # verknüpfe alle inputs mit den Variablen in der Berechnung
            dts : exTimes,
            K : strike,
            r : riskfree_r,
            sigma: impliedvol,
            delta: dividend,
            dW : random_train,
            train_weights : train_functions,
            part : N_train
        })

        # Starte Rückwärtsiteration um Fortführungswerte-Regression für jeden Ausübungszeitpunkt zu erhalten
        cont_value = np.zeros((N_train,1)) 
        
        for i in range(n_exercises-1)[::-1]:                                # starte beim vorletzen Ausübungszeitpunkt 
            y = exercise_values[:, i+1:i+2]
            X = paths[:, i]                                                    
            X_reg = feature_matrix(X, exercise_values[:, i:i+1], poly, N_train, d)
            
            checkpoint_path = "training_" + str(i) +"/cp.ckpt"
            checkpoint_dir = os.path.dirname(checkpoint_path)
            cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=1, save_weights_only=True, period=3)      
            model = build_model(X_reg)
            model.save_weights(checkpoint_path.format(epoch=0))
            
            history = model.fit(X_reg[exercise_values[:,i]>0], y[exercise_values[:,i]>0],
                                epochs = 100, 
                                validation_split = 0.2, 
                                verbose = 0,
                                callbacks = [cp_callback, early_stop, PrintDot()])
            
            train_functions.append(tf.train.latest_checkpoint(checkpoint_dir))
            cont_value = model.predict(X_reg)                         
            exercise_values[:, i:i+1] = np.maximum(exercise_values[:, i:i+1], cont_value)  
        
        # Forward simulation mit trainierten Gewichten w
        npv, greeks = sess.run([npv, greeks], { S: S_0,
                                                dts : exTimes,
                                                K : strike,
                                                r : riskfree_r,
                                                sigma: impliedvol,
                                                delta: dividend,
                                                dW : random_pricing,
                                                train_weights : train_functions,
                                                part : N_pricing
                                              })
        return(npv, greeks)
    

##############################################################
### AB HIER CODE FÜR BEPREISUNG MIT VERSTÄRKTER REGRESSION ###
##############################################################

def pricing_function_Reinf(number_call_dates, poly, N_train, N_pricing, d):
    S = tf.placeholder(tf.float32)
    dts = tf.placeholder(tf.float32)
    K = tf.placeholder(tf.float32)
    r = tf.placeholder(tf.float32)
    sigma = tf.placeholder(tf.float32)
    delta = tf.placeholder(tf.float32)
    dW = tf.placeholder(tf.float32)
    train_weights = tf.placeholder(tf.float32)
    part = tf.placeholder(tf.int32)
    
    for j in range(d):
        vola = sigma[j]
        dW_j = dW[j*part:(j+1)*part,:]
        
        S_t_temp = S * tf.cumprod(tf.exp((r-delta-vola**2/2)*dts + vola*tf.sqrt(dts)*dW_j), axis=1)
        E_t_temp = tf.exp(-r*tf.cumsum(dts))*tf.maximum(S_t_temp-K, 0)
            
        if j == 0:
            S_t = S_t_temp
            E_t = E_t_temp
        else:
            S_t = tf.concat([S_t, S_t_temp], axis=0, name='concatS_t')
            E_t = tf.maximum(E_t, E_t_temp, name='Max2')      
    
    
    continuationValues = []
    exercises = [] 
    previous_exercises = 0
    npv = 0
    for i in range(number_call_dates-1):
        w = train_weights[-i]
        if i == 0 or i == number_call_dates-2:
            X = feature_matrix_from_current_state(S_t[:,i], E_t[:,i], poly, N_pricing, d, 0)
        else:
            X = feature_matrix_from_current_state(S_t[:,i], E_t[:,i], poly, N_pricing, d, w)
        contValue = tf.matmul(X, w)
        continuationValues.append(contValue)
        inMoney = tf.cast(tf.greater(E_t[:,i], 0.), tf.float32)
        exercise = tf.cast(tf.greater(E_t[:,i], contValue[:,0]), tf.float32) * inMoney 
        exercises.append(exercise)
        exercise = exercise * (1-previous_exercises)
        previous_exercises += exercise
        npv += exercise*E_t[:,i]
    
    # Last exercise date
    inMoney = tf.cast(tf.greater(E_t[:,-1], 0.), tf.float32)          # Underlying-Werte > 0 z.Z. t=T -> 1, ansonsten 0
    exercise =  inMoney * (1-previous_exercises)                      # Wenn zuvor schon ausgeübt -> 0, ansonsten 1
    npv += exercise*E_t[:,-1]                                         # Cashflow in t=T * Ausübungen in t=T, addiere zum npv
    npv = tf.reduce_mean(npv)                                         # Mittelwert über alle Ausübungswerte/Cashflows
    greeks = tf.gradients(npv, [S, r, sigma])                         # Greeks = Ableitung von npv in einzelne Variablen
    return([S, dts, K, r, sigma, delta, dW, S_t, E_t, npv, greeks, train_weights, part])

def bermudanMC_tensorFlow_Reinf(S_0, strike, T, calldates, dividend, impliedvol, riskfree_r, random_train, random_pricing, plots, poly):
    d = len(impliedvol)
    N_pricing = int(len(random_pricing)/d)
    N_train = int(len(random_train)/d)
    exTimes = []
    dist = T/calldates
    k = 1
    while k <= calldates:                                       # Erstelle Liste mit äquidistanten Ausübungszeitpunkten
        exTimes.append(dist)
        k = k + 1
    
    n_exercises = len(exTimes)
    train_functions = []
    
    with tf.Session() as sess:                                  # eröffne Berechnungsumgebung (session)
        # berechne Pfade und npv´s zu jeder Zeit t
        S, dts, K, r, sigma, delta, dW, S_t, E_t, npv, greeks, train_weights, part = pricing_function_Reinf(n_exercises, poly, N_train, N_pricing, d)
        sess.run(tf.global_variables_initializer())             # initialisiere ops 
        paths, exercise_values = sess.run([S_t, E_t], {  
            S: S_0,                                             # verknüpfe alle inputs mit den Variablen in der Berechnung
            dts : exTimes,
            K : strike,
            r : riskfree_r,
            sigma: impliedvol,
            delta: dividend,
            dW : random_train,
            train_weights : train_functions,
            part : N_train
        })

        # Starte Rückwärtsiteration um Fortführungswerte für jeden Ausübungszeitpunkt zu erhalten
        cont_value = np.zeros((N_train,1)) 
        
        for i in range(n_exercises-1)[::-1]:                                # starte beim vorletzen Ausübungszeitpunkt 
            y = exercise_values[:, i+1:i+2]
            X = paths[:, i]                                                 # Underlying Werte zum Zeitpunkt
            X_reg = feature_matrix(X, exercise_values[:, i:i+1], poly, N_train, d)
            V = np.maximum(y, cont_value)
            
            if i == n_exercises-3:
                boost = np.maximum(y, np.matmul(X_reg,w))
                X_reg = np.c_[X_reg, boost]
                
            if i < n_exercises-3:
                X_reg = np.c_[X_reg, boost]
                boost = np.maximum(y, np.matmul(X_reg,w))
                X_reg = np.c_[X_reg[:,0:-1], boost]
            
            X_trans = np.transpose(X_reg[exercise_values[:,i]>0])                               # nur in the money pfade benutzen
            w = np.matmul(np.matmul(np.linalg.inv(np.matmul(X_trans,X_reg[exercise_values[:,i]>0])),X_trans),V[exercise_values[:,i]>0]) 
            train_functions.append(w)
            cont_value = np.matmul(X_reg,w)
            exercise_values[:, i:i+1] = np.maximum(exercise_values[:, i:i+1], cont_value)   # Ausübungspreis berechnen
        
        # Forward simulation mit trainierten Gewichten w
        npv, greeks = sess.run([npv, greeks], { S: S_0,
                                                dts : exTimes,
                                                K : strike,
                                                r : riskfree_r,
                                                sigma: impliedvol,
                                                delta: dividend,
                                                dW : random_pricing,
                                                train_weights : train_functions,
                                                part : N_pricing
                                              })
        return(npv, greeks)
    
############################################################ ab hier GUI Code

class Window(Frame):
 
    def __init__(self, master=None):
        Frame.__init__(self, master)    
        self.master = master
        self.init_window()
        
    #Creation of init_window
    def init_window(self):   
        self.master.title("Pricing Tool")
        self.pack(fill=BOTH, expand=1) # allowing the widget to take the full space of the root window 
        
        # Defining Buttons
        quitButton = Button(self, text="Exit", command=self.client_exit, height = 1, width = 4)
        quitButton.place(x=400, y=410)
        refreshButton = Button(self, text="Refresh", command=self.refresh_values, height = 1, width = 8)
        refreshButton.place(x=110, y=410)
        DefaultButton = Button(self, text="Set to Default", command=self.default_values, height = 1, width = 10)
        DefaultButton.place(x=20, y=410)       
        CalcButton = Button(self, text="Calculate Price", command=self.calc_price, height = 1, width = 12)
        CalcButton.place(x=190, y=410)       
          
        # Dropdown Buttons
        TypeText = Label(self, text="Option type:")
        TypeText.place(x=300 ,y=60)
        TypeLib = ["Bermudan Max Call"]
        OptionType.set(TypeLib[0])  
        TypeDropdown = OptionMenu(self, OptionType, *TypeLib)
        TypeDropdown.place(x=300, y=80)
        
        PolyText = Label(self, text="Polynomial type:")
        PolyText.place(x=300 ,y=110)
        PolyLib = ["Laguerre","Chebychev","Legendre","Hermite","mixed","mixed_exc","mixed_squared","mixed_max","mixed_squared_exc"]
        Polynomials.set(PolyLib[0])
        PolyDropdown = OptionMenu(self, Polynomials, *PolyLib)
        PolyDropdown.place(x=300 ,y=130)
        
        AlgoText = Label(self, text="Calculation Type:")
        AlgoText.place(x=300 ,y=160)
        AlgoLib = ["Analytical","Neural Net","Reinforced"]
        Algorythm.set(AlgoLib[0])
        AlgoDropdown = OptionMenu(self, Algorythm, *AlgoLib)
        AlgoDropdown.place(x=300 ,y=180)        
            
        # Defining Entry and Textwindows for Parameters
        Headline = Label(self, text="Input Parameter", font=("Helvetica", 20))
        Headline.place(x=140, y=10)
        SecHeadline = Label(self, text="Option Price", font=("Helvetica", 20))
        SecHeadline.place(x=160, y=280)        
        
        SpotText = Label(self, text="Spot:")
        SpotText.place(x=20, y=60)
        SpotEntry = Entry(self, textvariable= Spot)
        SpotEntry.place(x=100, y=60)
        
        StrikeText = Label(self, text="Strike:")
        StrikeText.place(x=20, y=80)
        StrikeEntry = Entry(self, textvariable= Strike)
        StrikeEntry.place(x=100, y=80)
        
        MaturityText = Label(self, text="Maturity:")
        MaturityText.place(x=20, y=100)
        MaturityEntry = Entry(self, textvariable= Maturity)
        MaturityEntry.place(x=100, y=100)       
        
        CalldatesText = Label(self, text="Calldates:")
        CalldatesText.place(x=20, y=120)
        CalldatesEntry = Entry(self, textvariable= Calldates)
        CalldatesEntry.place(x=100, y=120)
        
        DividendText = Label(self, text="Dividend:")
        DividendText.place(x=20, y=140)
        DividendEntry = Entry(self, textvariable= Dividend)
        DividendEntry.place(x=100, y=140)     
        
        impliedVolText = Label(self, text="Implied Vola:")
        impliedVolText.place(x=20, y=160)
        impliedVolEntry = Entry(self, textvariable= impliedVol)
        impliedVolEntry.place(x=100, y=160)
        
        IRText = Label(self, text="Interest rate:")
        IRText.place(x=20, y=180)
        IREntry = Entry(self, textvariable= IR)
        IREntry.place(x=100, y=180) 
        
        DimText = Label(self, text="#Assets:")
        DimText.place(x=20, y=200)
        DimEntry = Entry(self, textvariable= Assets)
        DimEntry.place(x=100, y=200)  
        
        LearnSampleText = Label(self, text="Number of Learning Samples:")
        LearnSampleText.place(x=20, y=230)
        LearnSampleEntry = Entry(self, textvariable= LearnSample)
        LearnSampleEntry.place(x=200, y=230)        
        
        PriceSampleText = Label(self, text="Number of Pricing Samples:")
        PriceSampleText.place(x=20, y=250)
        PriceSampleEntry = Entry(self, textvariable= PriceSample)
        PriceSampleEntry.place(x=200, y=250)        
    
        LowerHead = Label(self, text="Lower Bound for Option Price:", font=("Helvetica", 10))
        LowerHead.place(x=20, y=330)
        UpperHead = Label(self, text="Upper Bound for Option Price:", font=("Helvetica", 10))
        UpperHead.place(x=20, y=350)
        TimeHead = Label(self, text="Used time for calculation (hh:mm:ss.ms):", font=("Helvetica", 10))
        TimeHead.place(x=20, y=370)    
        
        LowerEntry = Entry(self, textvariable= Result_low)
        LowerEntry.place(x=280, y=330)
        UpperEntry = Entry(self, textvariable= Result_up)
        UpperEntry.place(x=280, y=350)       
        TimeEntry = Entry(self, textvariable= TimeUsed)
        TimeEntry.place(x=280, y=370)
    
    
    def default_values(self):
        Spot.set(100.)
        Strike.set(100.)
        Maturity.set(3.)
        Calldates.set(9)
        Dividend.set(0.1)
        impliedVol.set(0.2)
        IR.set(0.05)
        Assets.set(2)
        LearnSample.set(10000)
        PriceSample.set(10000)
        
    def refresh_values(self):
        Spot.set(0.)
        Strike.set(0.)
        Maturity.set(0.)
        Calldates.set(0)
        Dividend.set(0.)
        impliedVol.set(0.)
        IR.set(0.)
        Assets.set(0)
        LearnSample.set(0)
        PriceSample.set(0)           
    
    
    def calc_price(self):
        start_time = dt.datetime.now() 
        np.random.seed(42)      
        N = np.random.randn(LearnSample.get(), Calldates.get())
        N_pricing = np.random.randn(PriceSample.get(), Calldates.get())
        vola = []
        for k in range(1,Assets.get()+1):
            vola.append(impliedVol.get())       
        Method = Algorythm.get()
        if Method == "Analytical":
            npv, greeks = bermudanMC_tensorFlow(Spot.get(),Strike.get(),Maturity.get(),Calldates.get(),Dividend.get(),vola,IR.get(),N,N_pricing,'No',Polynomials.get())

        elif Method == "Neural Net":
            npv, greeks = bermudanMC_tensorFlow_NN(Spot.get(),Strike.get(),Maturity.get(),Calldates.get(),Dividend.get(),vola,IR.get(),N,N_pricing,'No',Polynomials.get())

        elif Method == "Reinforced":
            npv, greeks = bermudanMC_tensorFlow_Reinf(Spot.get(),Strike.get(),Maturity.get(),Calldates.get(),Dividend.get(),vola,IR.get(),N,N_pricing,'No',Polynomials.get())   

        if Polynomials.get() == "Laguerre":
            A, upper = bermudanMC_tensorFlow_wUB(Spot.get(),Strike.get(),Maturity.get(),Calldates.get(),Dividend.get(),vola,IR.get(),N,N_pricing,'No','Legendre')    
        else:    
            A, upper = bermudanMC_tensorFlow_wUB(Spot.get(),Strike.get(),Maturity.get(),Calldates.get(),Dividend.get(),vola,IR.get(),N,N_pricing,'No',Polynomials.get())
        
        time_elapsed = dt.datetime.now() - start_time
        time_elapsed_asString = str(time_elapsed)        
        Result_low.set(round(npv,5))
        Result_up.set(round(upper,5))
        TimeUsed.set(time_elapsed_asString)
        
        
    def client_exit(self):
        root.destroy()   
        
           
root = Tk()
root.geometry("460x450")

OptionType = StringVar()
Polynomials = StringVar()
Algorythm = StringVar()
Spot = DoubleVar()
Strike = DoubleVar()            #Options: StringVar(),  IntVar(),   DoubleVar(),  BooleanVar()
Maturity = DoubleVar()
Calldates = IntVar()
Dividend = DoubleVar()
impliedVol = DoubleVar()
IR = DoubleVar()
Assets = IntVar()
LearnSample = IntVar()
PriceSample = IntVar()
Result_low = DoubleVar()
Result_up = DoubleVar()
TimeUsed = StringVar()
    
app = Window(root)
try:
    root.tk.call('wm', 'iconphoto', root._w, PhotoImage(file='GUIIcon.png'))
except TclError:
    pass
root.mainloop()
