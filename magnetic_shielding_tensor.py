import pandas as pd
import numpy as np
import itertools as it
import statistics as stat

def mst_matrix(index):
    matrix = np.array([[mag_shield_tensorData.iloc[index, 2], mag_shield_tensorData.iloc[index, 3], mag_shield_tensorData.iloc[index, 4]],
                    [mag_shield_tensorData.iloc[index, 5], mag_shield_tensorData.iloc[index, 6], mag_shield_tensorData.iloc[index, 7]],
                    [mag_shield_tensorData.iloc[index, 8], mag_shield_tensorData.iloc[index, 9], mag_shield_tensorData.iloc[index, 10]]])
    return matrix

def commutator(m1, m2):
    return (np.matmul(m1, m2)-np.matmul(m2, m1))

def anticommutator(m1, m2):
    return (np.matmul(m1, m2)+np.matmul(m2, m1))

def eigvec_dot(m1, m2):
    eigval1, eigvec1=np.linalg.eig(m1)
    eigval2, eigvec2=np.linalg.eig(m2)

    eigvec_dot_list=[]
    for i, j in it.product(range(3), range(3)):
        eigvec_dot_list.append([np.dot(eigvec1[i], eigvec2[j])])

    return np.array(eigvec_dot_list)




trainDataLocation = "C:\\Users\\lawre\\Downloads\\train.csv (1)\\train.csv"
mag_shield_tensor_location='C:\\Users\\lawre\\Downloads\\magnetic_shielding_tensors.csv\\magnetic_shielding_tensors.csv'

mag_shield_tensorData=pd.read_csv(mag_shield_tensor_location, header=0, nrows=45)
trainDataRaw = pd.read_csv(trainDataLocation, header=0, nrows=82)
trainDataRaw['comm_eigvals1'] = ""
trainDataRaw['comm_eigvals2'] = ""
trainDataRaw['comm_eigvals3'] = ""
trainDataRaw['comm_traces']=""
trainDataRaw['comm_offdiagsums']=""
trainDataRaw['comm_deters']=""

trainDataRaw['anticomm_eigvals1'] = ""
trainDataRaw['anticomm_eigvals2'] = ""
trainDataRaw['anticomm_eigvals3'] = ""
trainDataRaw['anticomm_traces']=""
trainDataRaw['anticomm_offdiagsums']=""
trainDataRaw['anticomm_deters']=""

trainDataRaw['sum_eigvals1'] = ""
trainDataRaw['sum_eigvals2'] = ""
trainDataRaw['sum_eigvals3'] = ""
trainDataRaw['sum_traces']=""
trainDataRaw['sum_offdiagsums']=""
trainDataRaw['sum_deters']=""

trainDataRaw['eigvec_dotmin']=""
trainDataRaw['eigvec_dotmax']=""
trainDataRaw['eigvec_dotmed']=""

numAtoms = mag_shield_tensorData['molecule_name'].value_counts()
uniqueMolecules = mag_shield_tensorData['molecule_name'].value_counts().index.values.tolist()
uniqueMoleculeValues = []
for a, b in zip(uniqueMolecules, numAtoms):
    temp = int(a.split("_")[1])
    uniqueMoleculeValues.append([temp,a,b])

uniqueMoleculeValues.sort(key=lambda molecule: molecule[0])

idCounter=0
for entry in uniqueMoleculeValues:

    # TODO: Make separate entries; this will just put a list of bond distances in a single array per the first
    # entry for each molecule.

    # Create a list with the possible combinations.
    combList = list(it.combinations(range(0,entry[2]),2))
    combListInd = []
    for x in combList:
        if x[0]==0:
            x=x[::-1]
        combListInd.append(list(x))



    # # Start calculating bond lengths.
    # lengths = []
    for i in combListInd:
        trainDataInd=trainDataRaw.loc[(trainDataRaw['molecule_name']==mag_shield_tensorData['molecule_name'][idCounter])&
                            ((trainDataRaw['atom_index_0']==i[0]) | (trainDataRaw['atom_index_0']==i[1]))&
                            ((trainDataRaw['atom_index_1']==i[1]) | (trainDataRaw['atom_index_1']==i[0]))].index

        if len(trainDataInd)>0:
            matrix1=mst_matrix(i[0]+idCounter)
            matrix2=mst_matrix(i[1]+idCounter)

            comm_eigval=np.linalg.eigvals(commutator(matrix1, matrix2))
            comm_eigval.sort() #defaults to sorting by real component
            trainDataRaw.loc[trainDataInd[0], 'comm_eigvals1'] = comm_eigval[0]
            trainDataRaw.loc[trainDataInd[0], 'comm_eigvals2'] = comm_eigval[1]
            trainDataRaw.loc[trainDataInd[0], 'comm_eigvals3'] = comm_eigval[2]

            comm_trace=np.trace(commutator(matrix1, matrix2))
            comm_offdiagsum=np.sum(commutator(matrix1, matrix2))-comm_trace
            trainDataRaw.loc[trainDataInd[0], 'comm_traces'] = comm_trace if np.abs(comm_trace)>1e-11 else 0
            trainDataRaw.loc[trainDataInd[0], 'comm_offdiagsums'] = comm_offdiagsum if np.abs(comm_offdiagsum)>1e-11 else 0

            comm_deter=np.linalg.det(commutator(matrix1, matrix2))
            trainDataRaw.loc[trainDataInd[0], 'comm_deters'] = comm_deter if np.abs(comm_deter)>1e-11 else 0

            anticomm_eigval=np.linalg.eigvals(anticommutator(matrix1, matrix2))
            anticomm_eigval.sort() #defaults to sorting by real component
            trainDataRaw.loc[trainDataInd[0], 'anticomm_eigvals1'] = anticomm_eigval[0]
            trainDataRaw.loc[trainDataInd[0], 'anticomm_eigvals2'] = anticomm_eigval[1]
            trainDataRaw.loc[trainDataInd[0], 'anticomm_eigvals3'] = anticomm_eigval[2]

            anticomm_trace=np.trace(anticommutator(matrix1, matrix2))
            anticomm_offdiagsum=np.sum(anticommutator(matrix1, matrix2))-anticomm_trace
            trainDataRaw.loc[trainDataInd[0], 'anticomm_traces'] = anticomm_trace if np.abs(anticomm_trace)>1e-11 else 0
            trainDataRaw.loc[trainDataInd[0], 'anticomm_offdiagsums'] = anticomm_offdiagsum if np.abs(anticomm_offdiagsum)>1e-11 else 0

            anticomm_deter=np.linalg.det(anticommutator(matrix1, matrix2))
            trainDataRaw.loc[trainDataInd[0], 'anticomm_deters'] = anticomm_deter if np.abs(anticomm_deter)>1e-11 else 0

            sum_eigval=np.linalg.eigvals(matrix1+matrix2)
            sum_eigval.sort() #defaults to sorting by real component
            trainDataRaw.loc[trainDataInd[0], 'sum_eigvals1'] = sum_eigval[0]
            trainDataRaw.loc[trainDataInd[0], 'sum_eigvals2'] = sum_eigval[1]
            trainDataRaw.loc[trainDataInd[0], 'sum_eigvals3'] = sum_eigval[2]

            sum_trace=np.trace(matrix1+matrix2)
            sum_offdiagsum=np.sum(matrix1+matrix2)-sum_trace
            trainDataRaw.loc[trainDataInd[0], 'sum_traces'] = sum_trace if np.abs(sum_trace)>1e-11 else 0
            trainDataRaw.loc[trainDataInd[0], 'sum_offdiagsums'] = sum_offdiagsum if np.abs(sum_offdiagsum)>1e-11 else 0

            sum_deter=np.linalg.det(matrix1+matrix2)
            trainDataRaw.loc[trainDataInd[0], 'sum_deters'] = sum_deter if np.abs(sum_deter)>1e-11 else 0

            eigvector_dot=eigvec_dot(matrix1, matrix2)
            trainDataRaw.loc[trainDataInd[0], 'eigvec_dotmin'] = eigvector_dot.min()
            trainDataRaw.loc[trainDataInd[0], 'eigvec_dotmax'] = eigvector_dot.max()
            trainDataRaw.loc[trainDataInd[0], 'eigvec_dotmed'] = stat.median(eigvector_dot)


#
#     # Update index counter and store results in first index for each unique molecule.
#     structureDataRaw.iat[idCounter, 6] = lengths
    idCounter += entry[2]

print(trainDataRaw.head(20))
# one matrix scalar quantity: eigenvalue, trace, determinant, sum of off-diagonal components
# two matrices operations: commutator, commutator of transpose, diagonalize commutator, commutator of diagonalized matrix, sum, anticommutator
# two matrices sclar quantity: one matrix scalar quantities on two matrices operations, eigenvector dot products

