# non-linear_optimization_techniques
optimization by using the conventional gradient descent, Newton-Raphson and line search gradient descent (steepest) techniques


--Gradient Descent Method: is a first-order iterative optimization algorithm for finding a local minimum of a differentiable function. To find a local minimum of a function using gradient descent, we take steps proportional to the negative of the gradient (or approximate gradient) of the function at the current point. But if we instead take steps proportional to the positive of the gradient, we approach a local maximum of that function; the procedure is then known as gradient ascent.
loop
               ______________________________________
    |∇f(Xn)| =|(df/dx1)^2 + (df/dx2)^2 + (df/dx3)^2
    Gradient = [df/dx1]
               |df/dx2|
               [df/dx3]
                      3*1
    Xn+1 = Xn − η∇f(Xn)
    Xn = Xn+1

--Newton-Raphson Method: aims at estimating the roots of a function. For this purpose, an initial approximation is chosen, after this, the equation of the tangent line of the function at this point and the intersection of it with the axis of the abscissa, to find a better approximation for the root, is calculated.
By repeating the process, an iterative method is created to find the root of the function.
loop
    Gradient = [df/dx1]
               |df/dx2|
               [df/dx3]
                      3*1
    |∇f(Xn)| =|(df/dx1)^2 + (df/dx2)^2 + (df/dx3)^2
    Hessian = [d2f/dx1^2   d2f/dx1dx2  d2f/dx1dx3]
              |d2f/dx2dx1  d2f/dx2^2   d2f/dx2dx3|
              [d2f/dx3dx1  d2f/dx3dx2  d2f/dx3^2 ]
                                                 3*3
    Xn+1 = Xn − (H^(-1)) * f_dash
    Xn = Xn+1
--Steepest Gradient Descent Method: is a special case of gradient descent where the step length is chosen to minimize the objective function value. Gradient descent refers to any of a class of algorithms that calculate the gradient of the objective function, then move "downhill" in the indicated direction; the step length can be fixed, estimated by Newton-Raphson technique to get desired Minimum Step Length.
loop 
    Gradient = [df/dx1]
               |df/dx2|
               [df/dx3]
                     3*1
            ______________________________________
    |∇f(Xn)| =|(df/dx1)^2 + (df/dx2)^2 + (df/dx3)^2
    Xn+1 = Xn − η∇f(Xn)
    𝐹(η)=1/2[𝑔1(η)]^2+1/2[𝑔2(η))]^2+1/2[𝑔3(η)]^2
    loop
        ηn+1 = ηn _ η'/η``
        ηn = ηn+1 
    Xn+1 = Xn − η∇f(xn)
    Xn = Xn+1
