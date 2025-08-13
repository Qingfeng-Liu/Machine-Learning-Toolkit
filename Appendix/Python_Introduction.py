{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 1 Python Basics\n",
    "Jupyter Notebook reference: [https://www.javadrive.jp/python/jupyter-notebook/](https://www.javadrive.jp/python/jupyter-notebook/)  \n",
    "Installation and startup sections not needed (we'll launch via Anaconda)."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1.1 Hello World"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"Hello World\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Practice"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Combine output using + operator"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"Hello\" + \" \" + \"world\" + \" ^_^\" + \"6\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "x = 20\n",
    "print(x)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "x = 10\n",
    "print(x)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1.2 Arithmetic Operations"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "x = 10\n",
    "y = 20\n",
    "z = x - y\n",
    "s = x + y\n",
    "w = x * y\n",
    "m = x / y\n",
    "print(x, y, z, s, w, m)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Exponentiation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(2**2, 2**3, 2**5, 2**(1/3), 2**0.5, 2**(1/2))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "2**(0.5)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1.3 Data Types"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Integer (int) type"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "x = 10\n",
    "print(x)\n",
    "type(x)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Floating point (float) type"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "y = 10.2\n",
    "print(y)\n",
    "type(y)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### String (str) type"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "name = \"Liu Qingfeng\"\n",
    "print(name)\n",
    "type(name)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Boolean (bool) type: True and False"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "s0 = (8 > 5)\n",
    "print(s0)\n",
    "type(s0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "s1 = (5 > 3)\n",
    "print(s1)\n",
    "type(s1)\n",
    "print(s0*s1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### List type"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "l1 = [3, 5, 9]\n",
    "print(l1)\n",
    "type(l1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "l2 = ['A', 5, \"python\", \"0.01\",\"Class is over 你好\"]\n",
    "print(l2)\n",
    "type(l2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(l2[0])\n",
    "type(l2[0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "type(l2[1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "type(l2[2])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(l2[-3])\n",
    "type(l2[-3])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1.4 Introduction to Numpy Module"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np  # Abbreviated as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "x = np.array([2, 4, 9])  # Array method\n",
    "print(x)\n",
    "type(x)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Matrix Operations\n",
    "#### Reference: [https://note.nkmk.me/python-numpy-matrix/](https://note.nkmk.me/python-numpy-matrix/)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "x = np.array([[1, 2],[3,4]])\n",
    "print(f\"x = \\n{x}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "y = np.array([[10,100],[100,1]])\n",
    "print(f\"y = \\n{y}\")\n",
    "print(\"x * y = \")\n",
    "print(x * y)  # Element-wise multiplication\n",
    "print(x@y)    # Matrix multiplication\n",
    "print(np.dot(x,y))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "x = np.array([[1, 2],[10, 20],[300, 400]])\n",
    "print(x)\n",
    "x.shape  # Outputs the shape: 3 rows × 2 columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "y = np.array([[3, 4, 5, 6],[7, 8, 9, 5]])\n",
    "print(y, y.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "x*y  # Cannot multiply due to size mismatch"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "x@y  # Matrix multiplication"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Inverse Matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "x = np.array([[1, 2],[3,4]])\n",
    "xinv = np.linalg.inv(x)\n",
    "print(np.dot(x,xinv))  # Product of x and xinv\n",
    "print(x@xinv)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Determinant"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "np.linalg.det(x)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Eigenvalues and Eigenvectors"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "w, v = np.linalg.eig(x)\n",
    "print('Eigenvalues:', w)\n",
    "print('Eigenvectors:', v)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1.5 For Loops\n",
    "Display natural numbers from 1 to 10"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "for i in np.arange(10):\n",
    "    x = (i + 1)\n",
    "    print(x)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Sum numbers from 1 to 10"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "xsum = 0\n",
    "for i in np.arange(10):\n",
    "    xsum += (i + 1)\n",
    "    print(xsum)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Practice Problems\n",
    "1. Calculate the product sum of natural numbers from 1 to 10 using a for loop.\n",
    "2. Calculate the product sum of [2, 5, 1, 8] using a for loop."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1.6 If Conditions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Regular if statement example\n",
    "x = 8\n",
    "if x > 10:\n",
    "    print(\"x is greater than 10\")\n",
    "elif x > 7:\n",
    "    print(\"x is greater than 7\")\n",
    "else:\n",
    "    print(\"x is 7 or less\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1.7 Functions"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Sum calculation function"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def mysum(x):\n",
    "    xsum = 0\n",
    "    for i in np.arange(len(x)):\n",
    "        print(i)\n",
    "        xsum += x[i]\n",
    "    return xsum"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "x = [1,2,3,58,6,4,6,67,7,8,7]\n",
    "print(x)\n",
    "mysum(x)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "y = [20,30,40,50,60,0.1]\n",
    "z = mysum(y)\n",
    "print(z)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Practice Problem: \n",
    "The heights of 5 people are 150,160,165,180,190 cm, and their weights are 60,50,70,50,100 kg respectively. \n",
    "Find the standard weight formula online (height(m)^2 × 22), and if the 5 people are within ±3 kg of standard weight, \n",
    "judge as standard; otherwise obese or thin."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "x = np.array([[150,160,165,180,190],\n",
    "              [60,50,70,50,80]])\n",
    "StW = (x[0][:]/100)**2*22\n",
    "print(\"Standard Weights:\", StW)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print('Height:', x[0][1])\n",
    "print('Weights:', x[1][:])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "for i in np.arange(5):\n",
    "    if x[1][i] > StW[i]+3:\n",
    "        print(f'Person {i+1}: Obese')\n",
    "    elif x[1][i] < StW[i]-3:\n",
    "        print(f'Person {i+1}: Thin')\n",
    "    else:\n",
    "        print(f'Person {i+1}: Standard')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Pandas Introduction\n",
    "## Preparation: Upload DATA01.xls to the same folder (using upload button in left panel)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "!pip install openpyxl\n",
    "!pip install xlrd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df01 = pd.read_excel('DATA01.xls')\n",
    "df02 = pd.read_csv('CAPM02.csv', skiprows=1)\n",
    "print(\"DATA01 head:\")\n",
    "print(df01.head())\n",
    "print(\"\\nCAPM02 head:\")\n",
    "print(df02.head())\n",
    "X = df01[[\"Weight(kg)\"]]\n",
    "print(\"\\nWeight column:\")\n",
    "print(X)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df01.tail(1)  # Show last row"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df01.loc[0:3,'Weight(kg)']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df01.loc[0,:]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df01.loc[0:2,:]"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
