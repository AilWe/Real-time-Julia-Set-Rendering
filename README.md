# This is a CUDA/OpenGL interoperability project


## introduction

OpenGL is a graphics library with a wide range of APIs to render 2D and 3D graphics 
on the screen. It provides more than 200 functions for easily rendering vector graphics
on GPU, including primitive function, attribute functions, viewing functions, input 
functions, and control functions. CUDA is a parallel computing platform. It enables 
software developers to use GPU efficiently. CUDA-OpenGL Interoperability enables
the calculation of OpenGL can be done in parallel to achieve better performance. Even 
it is possible to finish all visualization without leaving GPU. 
## Implementation

Julia set is a set of complex numbers. The Julia set consists of values in which a small 
perturbation can cause drastic changes in the sequence of iterated function values [4]. 
The family of complex quadratic gives a very popular complex dynamical system. The 
Julia set can be expressed as the function:
`f(z) = z^2 +c`\
c is a complex number. For this iteration, the Julia set is a fractal. The calculation of 
Julia set is computing intensive when 𝑐𝑐 is updating. Thus, we use CUDA for the 
calculation.

$f(z) = z^2 + 0.578 * cos(a) + i* 0.578 * sin(a) $
The a has a range updated from 0 to 2* Pi.
The OpenGL is used to render the graphics. Majority calculation is finished on CUDA.

## Result
Run program:
`sh run realTimeJuliaSet`
`./realTimeJuliaSet`

The Julia Set with a range from 0 to 2Pi will be shown. 
