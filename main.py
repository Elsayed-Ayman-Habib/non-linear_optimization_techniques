import math as math

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import sympy as smp
import random 
from scipy.misc import derivative

x_1,x_2,x_3,η = smp.symbols('x_1 x_2 x_3 η', real=True)

#
# First Equation
# 𝑔1(𝑥1,𝑥2,𝑥3)=3𝑥1−cos(𝑥2𝑥3)−0.5
#
g_1 = (3 * x_1) - (smp.cos((x_2)*(x_3))) - (0.5)

#
# Second Equation
# 𝑔2(𝑥1,𝑥2,𝑥3)=𝑥1^2−81(𝑥2+0.1)^2+sin(𝑥3)+1.06
# 
g_2 = (x_1**2) - (81*(((x_2) + (0.1))**2)) + (smp.sin(x_3)) + (1.06)

#
# Third Equation
# 𝑔3(𝑥1,𝑥2,𝑥3)=exp(−𝑥1𝑥2)+20𝑥3+(10𝜋−3)/3
# 
g_3 = smp.exp(-((x_1)*(x_2))) + (20*x_3) + (((10*math.pi)-(3))/(3))

#
# 𝐹(𝑥1,𝑥2,𝑥3)=1/2[𝑔1(𝑥1,𝑥2,𝑥3)]^2+1/2[𝑔2(𝑥1,𝑥2,𝑥3)]^2+1/2[𝑔3(𝑥1,𝑥2,𝑥3)]^2
# 
f = (1/2)*(g_1**2) + (1/2)*(g_2**2) + (1/2)*(g_3**2)

#
# First Dervatives
# df/dx1
# df/dx2
# df/dx3
# 
dfdx_1 = smp.diff(f, x_1) 
dfdx_2 = smp.diff(f, x_2) 
dfdx_3 = smp.diff(f, x_3) 

#
# Gradiant Vector = [df/dx1]
#                   |df/dx2|
#                   [df/dx3]
#                          3*1
# 
Gradiant_Vector = np.matrix(
                            [
                                [dfdx_1],
                                [dfdx_2],
                                [dfdx_3]
                            ]
)

#
# second Dervatives
# d2f/dx1^2
# d2f/dx2^2
# d2f/dx3^2
# 
df2dx2_1 = smp.diff(f, x_1 , 2)
df2dx2_2 = smp.diff(f, x_2 , 2)
df2dx2_3 = smp.diff(f, x_3 , 2)

#
# second Dervatives
# d2f/dx1dx2
# d2f/dx1dx3
# 
df2dx_1dx_2 = smp.diff(dfdx_1, x_2)
df2dx_1dx_3 = smp.diff(dfdx_1, x_3)

#
# second Dervatives
# d2f/dx2dx1
# d2f/dx2dx3
# 
df2dx_2dx_1 = smp.diff(dfdx_2, x_1)
df2dx_2dx_3 = smp.diff(dfdx_2, x_3)

#
# second Dervatives
# d2f/dx3dx1
# d2f/dx3dx2
# 
df2dx_3dx_1 = smp.diff(dfdx_3, x_1)
df2dx_3dx_2 = smp.diff(dfdx_3, x_2)

#
# Hessian matrix of the function f
# 
H = np.matrix(
    [
        [df2dx2_1,df2dx_1dx_2,df2dx_1dx_3],
        [df2dx_2dx_1,df2dx2_2,df2dx_2dx_3],
        [df2dx_3dx_1,df2dx_3dx_2,df2dx2_3]
    ]
)

#
# You must choose the algorithm according to the following
# 1- gradient descent
# 2- Newton-Raphson
# 3- steepest gradient descent
# default in the Choose_Algorism Variable is (gradient descent)
#
Choose_Algorism = "steepest gradient descent"

iterations = 100
for i in range(iterations):
    arr_fplot = np.array([])
    arr_Aplot = np.array([])
    arr_xplot = np.array([])
    arr_yplot = np.array([])
    arr_zplot = np.array([])
    arr_Xplot = np.array([])
    arr_Yplot = np.array([])
    arr_Zplot = np.array([])
    #
    # optimize the above function using the conventional gradient descent technique.
    # 
    if Choose_Algorism == "gradient descent" :
        learning_rate = 0.0001
        Stop = 0.01
        Amp_df = 0.1
        iteration = 0
        #
        # Xn+1
        # 
        current = ([[0],
                    [0],
                    [0]])
        #
        # Xn
        # 
        previous = ([[random.randint(0, 10)],
                    [random.randint(0, 1)],
                    [random.randint(0, 19)]])
        print("Initial Point equal " , previous)
        while Amp_df > Stop :
            iteration += 1
            f_Value = float(f.subs([(x_1 , previous[0][0]) , (x_2 , previous[1][0]) , (x_3 , previous[2][0])]))
            dx_1 = int(dfdx_1.subs([(x_1 , previous[0][0]) , (x_2 , previous[1][0]) , (x_3 , previous[2][0])]))
            dx_2 = int(dfdx_2.subs([(x_1 , previous[0][0]) , (x_2 , previous[1][0]) , (x_3 , previous[2][0])]))
            dx_3 = int(dfdx_3.subs([(x_1 , previous[0][0]) , (x_2 , previous[1][0]) , (x_3 , previous[2][0])]))

            #
            #            ______________________________________
            # |∇f(Xn)| =|(df/dx1)^2 + (df/dx2)^2 + (df/dx3)^2
            # 
            Amp_df = math.sqrt((((dx_1)**2)+((dx_2)**2)+((dx_3)**2)))
            #
            # g_df = [df/dx1]
            #        |df/dx2|
            #        [df/dx3]
            #               3*1
            # 
            g_df = ([[dx_1],
                    [dx_2],
                    [dx_3]])
            #
            # Xn+1 = Xn − η∇f(Xn)
            # 
            current = np.subtract(previous,(np.multiply(learning_rate,g_df)))
            arr_fplot = np.append(arr_fplot , f_Value)
            arr_Aplot = np.append(arr_Aplot , Amp_df)

            arr_xplot = np.append(arr_xplot , previous[0][0])
            arr_yplot = np.append(arr_yplot , previous[1][0])
            arr_zplot = np.append(arr_zplot , previous[2][0])

            arr_Xplot = np.append(arr_Xplot , previous[0][0])
            arr_Yplot = np.append(arr_Yplot , previous[1][0])
            arr_Zplot = np.append(arr_Zplot , previous[2][0])

            #
            # Xn = Xn+1
            # 
            previous = np.array(current)
            print (Amp_df)

    elif Choose_Algorism == "Newton-Raphson" :
        iteration = 0
        Stop = 0.01
        f_dash_Amp = 0.02

        current = np.array([[0.00],
                            [0.00],
                            [0.00]])
        previous = ([[random.randint(0, 10)],
                    [random.randint(0, 1)],
                    [random.randint(0, 19)]])
        print("Initial Point equal " , previous)
        while f_dash_Amp > Stop :
            iteration += 1
            f_Value = float(f.subs([(x_1 , previous[0][0]) , (x_2 , previous[1][0]) , (x_3 , previous[2][0])]))
            dx_1 = int(dfdx_1.subs([(x_1 , previous[0][0]) , (x_2 , previous[1][0]) , (x_3 , previous[2][0])]))
            dx_2 = int(dfdx_2.subs([(x_1 , previous[0][0]) , (x_2 , previous[1][0]) , (x_3 , previous[2][0])]))
            dx_3 = int(dfdx_3.subs([(x_1 , previous[0][0]) , (x_2 , previous[1][0]) , (x_3 , previous[2][0])]))

            #
            # f_dash = [df/dx1]
            #          |df/dx2|
            #          [df/dx3]
            #                 3*1
            f_dash = ([[dx_1],
                    [dx_2],
                    [dx_3]])
            
            #            ______________________________________
            # |∇f(Xn)| =|(df/dx1)^2 + (df/dx2)^2 + (df/dx3)^2
            # 
            f_dash_Amp = math.sqrt((((dx_1)**2)+((dx_2)**2)+((dx_3)**2)))

            dx1_2  = int(df2dx2_1.subs([(x_1 , previous[0][0]) , (x_2 , previous[1][0]) , (x_3 , previous[2][0])]))
            dx1_x2 = int(df2dx_1dx_2.subs([(x_1 , previous[0][0]) , (x_2 , previous[1][0]) , (x_3 , previous[2][0])]))
            dx1_x3 = int(df2dx_1dx_3.subs([(x_1 , previous[0][0]) , (x_2 , previous[1][0]) , (x_3 , previous[2][0])]))

            dx2_2  = int(df2dx2_2.subs([(x_1 , previous[0][0]) , (x_2 , previous[1][0]) , (x_3 , previous[2][0])]))
            dx2_x1 = int(df2dx_2dx_1.subs([(x_1 , previous[0][0]) , (x_2 , previous[1][0]) , (x_3 , previous[2][0])]))
            dx2_x3 = int(df2dx_2dx_3.subs([(x_1 , previous[0][0]) , (x_2 , previous[1][0]) , (x_3 , previous[2][0])]))

            dx3_2  = int(df2dx2_3.subs([(x_1 , previous[0][0]) , (x_2 , previous[1][0]) , (x_3 , previous[2][0])]))
            dx3_x1 = int(df2dx_3dx_1.subs([(x_1 , previous[0][0]) , (x_2 , previous[1][0]) , (x_3 , previous[2][0])]))
            dx3_x2 = int(df2dx_3dx_2.subs([(x_1 , previous[0][0]) , (x_2 , previous[1][0]) , (x_3 , previous[2][0])]))

            #
            # H = [d2f/dx1^2   d2f/dx1dx2  d2f/dx1dx3]
            #     |d2f/dx2dx1  d2f/dx2^2   d2f/dx2dx3|
            #     [d2f/dx3dx1  d2f/dx3dx2  d2f/dx3^2 ]
            #                                         3*3
            f_double_dash = np.matrix(
                            [
                                [dx1_2,dx1_x2,dx1_x3],
                                [dx2_x1,dx2_2,dx2_x3],
                                [dx3_x1,dx3_x2,dx3_2]
                            ]
            )

            #
            # Invese of H
            # 
            f_double_dash_inv = np.linalg.inv(f_double_dash) 

            #
            # Xn+1 = Xn − (1/H) * f_dash
            # 
            current = np.subtract(previous,(np.dot(f_double_dash_inv,f_dash)))
            arr_fplot = np.append(arr_fplot , f_Value)
            arr_Aplot = np.append(arr_Aplot , f_dash_Amp)

            arr_xplot = np.append(arr_xplot , previous[0][0])
            arr_yplot = np.append(arr_yplot , previous[1][0])
            arr_zplot = np.append(arr_zplot , previous[2][0])

            arr_Xplot = np.append(arr_Xplot , previous[0][0])
            arr_Yplot = np.append(arr_Yplot , previous[1][0])
            arr_Zplot = np.append(arr_Zplot , previous[2][0])
            previous = np.array(current)
            print (f_dash_Amp)

    elif Choose_Algorism == "steepest gradient descent" :
        iteration = 0
        Stop = 0.01
        Stop_η = 0.01 
        Amp_df = 0.1
        current_η = 0.00
        previous_η = 0.00
        η_dash = 0.02
        η_double_dash = 0.02
        current = ([[0.00],
                    [0.00],
                    [0.00]])
        previous = ([[random.randint(0, 10)],
                    [random.randint(0, 1)],
                    [random.randint(0, 19)]])
        print("Initial Point equal " , previous)
        while Amp_df > Stop:
            iteration += 1
            f_Value = float(f.subs([(x_1 , previous[0][0]) , (x_2 , previous[1][0]) , (x_3 , previous[2][0])]))
            dx_1 = int(dfdx_1.subs([(x_1 , previous[0][0]) , (x_2 , previous[1][0]) , (x_3 , previous[2][0])]))
            dx_2 = int(dfdx_2.subs([(x_1 , previous[0][0]) , (x_2 , previous[1][0]) , (x_3 , previous[2][0])]))
            dx_3 = int(dfdx_3.subs([(x_1 , previous[0][0]) , (x_2 , previous[1][0]) , (x_3 , previous[2][0])]))
            
            #
            # g_df = [df/dx1]
            #        |df/dx2|
            #        [df/dx3]
            #               3*1
            # 
            g_df = ([[dx_1],
                    [dx_2],
                    [dx_3]])
            
            #            ______________________________________
            # |∇f(Xn)| =|(df/dx1)^2 + (df/dx2)^2 + (df/dx3)^2
            # 
            Amp_df = math.sqrt((((dx_1)**2)+((dx_2)**2)+((dx_3)**2)))

            #
            # Xn+1 = Xn − η∇f(Xn)
            # 
            current = np.subtract(previous,(np.multiply(η,g_df)))

            g1_η = g_1.subs([(x_1 , current[0][0]) , (x_2 , current[1][0]) , (x_3 , current[2][0])])
            g2_η = g_2.subs([(x_1 , current[0][0]) , (x_2 , current[1][0]) , (x_3 , current[2][0])])
            g3_η = g_3.subs([(x_1 , current[0][0]) , (x_2 , current[1][0]) , (x_3 , current[2][0])])

            #
            # 𝐹(η)=1/2[𝑔1(η)]^2+1/2[𝑔2(η))]^2+1/2[𝑔3(η)]^2
            # 
            f_η = (1/2)*(g1_η**2) + (1/2)*(g2_η**2) + (1/2)*(g3_η**2)

            df_dη = smp.diff(f_η, η) 
            df2dη2 = smp.diff(f_η, η , 2) 

            if iteration >1:
                η_dash = int(df_dη.subs(η , previous_η))

            while abs(η_dash) > Stop_η :
                η_dash = int(df_dη.subs(η , previous_η))
                η_double_dash = int(df2dη2.subs(η , previous_η))

                #
                # ηn+1 = ηn _ η'/η``
                # 
                current_η = previous_η - (η_dash/η_double_dash)
                previous_η = current_η
            
            #
            # Xn+1 = Xn − η∇f(xn)
            #
            current = np.subtract(previous,(np.multiply(current_η,g_df)))
            arr_fplot = np.append(arr_fplot , f_Value)
            arr_Aplot = np.append(arr_Aplot , Amp_df)

            arr_xplot = np.append(arr_xplot , previous[0][0])
            arr_yplot = np.append(arr_yplot , previous[1][0])
            arr_zplot = np.append(arr_zplot , previous[2][0])

            arr_Xplot = np.append(arr_Xplot , previous[0][0])
            arr_Yplot = np.append(arr_Yplot , previous[1][0])
            arr_Zplot = np.append(arr_Zplot , previous[2][0])
            previous = current
            print (Amp_df)


    else :
        print("You should choose an algorithm based on previous algorithms")
    
    if Choose_Algorism == "gradient descent" or Choose_Algorism == "Newton-Raphson" or Choose_Algorism == "steepest gradient descent" :
        print("Minimum Point equal " , previous)
        fx1 = range(iteration)
        fy1 = arr_fplot
        plt.plot(fx1,fy1)
        plt.xlabel("Number of iterations" ,fontsize='large', fontweight='bold')
        plt.ylabel("f(x)",fontsize='large', fontweight='bold')
        plt.show()
        Ax1 = range(iteration)
        Ay1 = arr_fplot
        plt.plot(Ax1,Ay1)
        plt.xlabel("Number of iterations",fontsize='large', fontweight='bold')
        plt.ylabel("Amplitude |∇f(Xn)|",fontsize='large', fontweight='bold')
        plt.show()   
        x1 = range(iteration)
        y1 = arr_xplot
        plt.plot(x1,y1)
        plt.xlabel("Number of iterations",fontsize='large', fontweight='bold')
        plt.ylabel("First Element of Xn",fontsize='large', fontweight='bold')
        plt.show()
        x2 = range(iteration)
        y2 = arr_yplot
        plt.plot(x2,y2)
        plt.xlabel("Number of iterations",fontsize='large', fontweight='bold')
        plt.ylabel("Second Element of Xn",fontsize='large', fontweight='bold')
        plt.show()
        x3 = range(iteration)
        y3 = arr_zplot
        plt.plot(x3,y3)
        plt.xlabel("Number of iterations",fontsize='large', fontweight='bold')
        plt.ylabel("Third Element of Xn",fontsize='large', fontweight='bold')
        plt.show()
        ax = plt.axes(projection="3d")
        ax.plot(arr_Xplot,arr_Yplot,arr_Zplot)
        ax.set_xlabel("X_1 of Xn",fontsize='large', fontweight='bold')
        ax.set_ylabel("X_2 of Xn",fontsize='large', fontweight='bold')
        plt.show()


