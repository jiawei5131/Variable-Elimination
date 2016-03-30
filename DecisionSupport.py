#Implement the function DecisionSupport

'''
For this question you may use the code from part 1

Note however that part 2 will be marked independent of part 1

The solution for VariableElimination.py will be used for testing part2 instead
of the copy you submit. 
'''

from MedicalBayesianNetwork import *
from VariableElimination import *

'''
Parameters:
             medicalNet - A MedicalBayesianNetwork object                        

             patient    - A Patient object
                          The patient to calculate treatment-outcome
                          probabilites for
Return:
         -A factor object

         This factor should be a probability table relating all possible
         Treatments to all possible outcomes
'''
def DecisionSupport(medicalNet, patient):
    # set the evidence according to the patient first
    #   Although these information are given, they don't show up in the final answer
    medicalNet.set_evidence_by_patient(patient)
    patientVars = patient.evidenceVariables()

    factors = medicalNet.net.factors()

    # restrict the factors containing patient's variables
    restrict_patientVars(factors, patientVars)

    queryVars = medicalNet.getOutcomeVars() + medicalNet.getTreatmentVars()
    givenVars = medicalNet.getTreatmentVars()

    return MedicalVE(factors, queryVars, givenVars)

'''
MedicalVE(net, queryVar)

 Parameters :
              net: a MedicalBayesianNetwork object
              queryVars: a list of Variable object
                        (the variables whose distributions we want to compute)

 Return:
         A distribution over the values of QueryVars
'''
def MedicalVE(factors, queryVars, givenVars):

    # calculate the fixed ordering of the variables that want to be eliminated
    orderedVars = MVE_min_fill_ordering(factors, queryVars)

    # eliminate undesired variables
    for var in orderedVars:
        rmFactors = []
        factorsCopy = list(factors)

        # Find the factors contain the var
        for i in range(0, len(factors)):
            # iterate over the copy of the original factors for modifying factors during iteration
            curFac = factorsCopy[i]
            scope = curFac.get_scope()
            if var in scope:
                rmFactors.append(curFac)
                factors.remove(curFac)

        toSumFac = multiply_factors(rmFactors)
        newFactor = sum_out_variable(toSumFac, var)
        factors.append(newFactor)

    finalFactor = multiply_factors(factors)

    # normalization
    MEV_normalizatioin(finalFactor, givenVars)

    return finalFactor

'''
restrict_patientVars(factor, variable, value):

Parameters :
              factor : the factor to restrict
              variable : the variable to restrict "factor" on
              value : the value to restrict to
Return:
              Void. This function only modifies factors that contain patientVars
'''
def restrict_patientVars(factors, patientVars):
    # restrict the factors contain patient's variables
    for i in range(0, len(factors)):
        for var in patientVars:
            if var in factors[i].get_scope():
                value = var.get_evidence()
                factors[i] = restrict_factor(factors[i], var, value)

'''
min_fill_ordering(Factors, QueryVar):

Parameters :
              Factors : the factors to search
              QueryVars: list of query variables
Return:
              A list of ordered variables that give the min fill ordering
'''
def MVE_min_fill_ordering(Factors, QueryVars):
    scopes = []
    for f in Factors:
        scopes.append(list(f.get_scope()))
    Vars = []
    for s in scopes:
        for v in s:
            if not v in Vars and v not in QueryVars:
                Vars.append(v)

    ordering = []
    while Vars:
        (var,new_scope) = min_fill_var(scopes,Vars)
        ordering.append(var)
        if var in Vars:
            Vars.remove(var)
        scopes = remove_var(var, new_scope, scopes)
    return ordering

'''
MEV_normalizatioin(finalFactor, givenVars):

Parameters :
              finalFactor : the factor needs to be normalized
              givenVars: a list of variables that arr said to be given
Return:
              Void. This function only modifies the input factor
'''
def MEV_normalizatioin(finalFactor, givenVars):
    # normalization
    tVarsFound = []
    # find the index of the given variables in the assignment
    tVarsIndices = []
    sumsList = [0]
    for tvar in givenVars:
        tVarsIndices.append(finalFactor.get_scope().index(tvar))
        sumsList *= len(tvar.domain())

    # find the sum of probabilities over the same given variables
    for assignment in finalFactor.get_assignment_iterator():
        # current treatment combination in the assignment
        curComb = helper_curComb(assignment, tVarsIndices)

        value = finalFactor.get_value(assignment)
        if curComb not in tVarsFound:
            tVarsFound.append(curComb)
            sumsList[tVarsFound.index(curComb)] = value
        else:
            sumsList[tVarsFound.index(curComb)] += value

    # calculate the normalized probabilities
    for assignment in finalFactor.get_assignment_iterator():
        curComb = helper_curComb(assignment, tVarsIndices)

        newValue = finalFactor.get_value(assignment) / sumsList[tVarsFound.index(curComb)]
        finalFactor.add_value_at_assignment(newValue, assignment)

'''
helper_curComb(curAssignment, tVarsIndices):

Parameters :
              curAssignment : current assignment
              tVarsIndices: a list of integers that indicate the position of the given
                variables in the assignment
Return:
              a list of value of given variables
'''
def helper_curComb(curAssignment, tVarsIndices):
    curComb = []
    for tvarIndex in tVarsIndices:
        curComb.append(curAssignment[tvarIndex])
    return curComb