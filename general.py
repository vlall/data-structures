#Hamming Distance
def hamming_distance(s1, s2):
  #Return the Hamming distance between equal-length sequences
  if len(s1) != len(s2):
      raise ValueError("Undefined for sequences of unequal length")
  return sum(ch1 != ch2 for ch1, ch2 in zip(s1, s2))

def zip_list(A, B):
  #A = [1,2,3,4,5,6,7,8,9]
  #B = ["A","B","C"]
  from itertools import cycle
  zip_list = zip(A, cycle(B)) if len(A) > len(B) else zip(cycle(A), B)
  return zip_list

#There's a function for that in SciPy, it's called Euclidean
def scipy_euclidean(a,b,):
  from scipy.spatial import distance
  a = (1,2,3)
  b = (4,5,6)
  dst = distance.euclidean(a,b)	
  return(dst)

def sim_pearson(prefs,p1,p2):
  # Get the list of mutually rated items
  si={}
  for item in prefs[p1]:
   if item in prefs[p2]: si[item]=1
  # Find the number of elements
  n=len(si)
  # if they are no ratings in common, return 0
  if n==0: return 0
  # Add up all the preferences
  sum1=sum([prefs[p1][it] for it in si])
  sum2=sum([prefs[p2][it] for it in si])
  # Sum up the squares
  sum1Sq=sum([pow(prefs[p1][it],2) for it in si])
  sum2Sq=sum([pow(prefs[p2][it],2) for it in si])
  # Sum up the products
  pSum=sum([prefs[p1][it]*prefs[p2][it] for it in si])
  # Calculate Pearson score
  num=pSum-(sum1*sum2/n)
  den=sqrt((sum1Sq-pow(sum1,2)/n)*(sum2Sq-pow(sum2,2)/n))
  if den==0: return 0
  r=num/den
  return r

                return float("inf")

        def __str__(self):
            return '\n'.join([str(i+1)+' '+str(layer) for i,layer in enumerate(self.layers)])
