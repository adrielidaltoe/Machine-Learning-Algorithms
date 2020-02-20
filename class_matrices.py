
# coding: utf-8

# In[39]:


class Matrix:
    
    ''' Calculates determinant and sum and product of 2 matrices. 
        Inform a matrix = [[l1],[l2],...,[ln]] '''
    
    def __init__(self, matrix):
        self.matrix = matrix
        
    def sumMatrix(self, other_matrix): #matrix sum
        try:
            result_sum = []
            for i in range(len(self.matrix)):
                a = list(map(lambda x,y: x+y, self.matrix[i], other_matrix.matrix[i]))
                result_sum.append(a)
            return result_sum
        except:
            print('The matrices must have the same type mxn')
        
    
    def productMatrix(self, outra_matrix):
        if len(self.matrix) == len(outra_matrix.matrix[0]):
            cols = len(outra_matrix.matrix[0])
            l = [[row[i] for row in outra_matrix.matrix] for i in range(cols)] #transposta da outra_matriz.matrix
            result_multiplica = []
        
            for i in range(len(self.matrix)):
                result_linha = [] #stores the result of the multiplication between self.matrix[i], l[j]
                for j in range(len(self.matrix)):
                    a = sum(list(map(lambda x,y: x*y, self.matrix[i], l[j])))
                    result_linha.append(a)
                result_multiplica.append(result_linha)
            return result_multiplica
        else:
            print('The number of rows of one matrix must be equals to the number columns of the other.')
    
    # Applying the Sarrus rule to calculate the determinant
    
    def _expandeMatriz(self): #matrix used to calculate determinant
        if len(self.matrix) == 2:
            return self.matrix
        else:
            matrix_expandida = self.matrix[:] 
            for i in range(len(self.matrix)-1): 
                matrix_expandida.append(self.matrix[i])
            return matrix_expandida
         
    def _inverterMatriz(self): #inverts matrix rows for secondary diagonal calculations
        inverse = []
        if len(self.matrix) < 3:
            for lista in self.matrix:
                inverse.append(list(reversed(lista)))
        else:
            for lista in self._expandeMatriz():
                inverse.append(list(reversed(lista)))
        return inverse
    
    def _productDiagonal(self, matrix): #self._expandeMatriz() or self._inverterMatriz()
    
        determinant = 0

        if len(self.matrix) == 2:
            produto_diagonal = 1
            for i in range(len(self.matrix)):
                produto_diagonal *= matrix[i][i]
            determinant += produto_diagonal
        else:
            for i in range(len(self.matrix)):
                k = 0
                k2 = 0
                produto_diagonal = 1
                for j in range(len(self.matrix)):
                    k1 = i
                    k = k1 + k2
                    produto_diagonal *= matrix[k][j]
                    k2 += 1
                determinant += produto_diagonal
        return determinant
    
    def findDeterminant(self):
        if len(self.matrix) == len(self.matrix[0]):
            determinant = self._productDiagonal(self._expandeMatriz()) - self._productDiagonal(self._inverterMatriz())
            return determinant
        else:
            print('There only exists determinant of square matrices')

