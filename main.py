
import wisardpkg as wp

X = [
      [1,1,1,0,2,0,0,0],
      [1,2,1,1,0,0,0,0],
      [0,0,0,0,1,2,1,1],
      [0,0,2,0,0,1,1,1]
    ]

X2 = [
    [0,0,0,0,0,1,1,1],
    [1,1,1,0,0,0,0,0],
    [1,1,1,1,0,0,0,0],
    [0,0,0,0,1,1,1,1],
    [0,0,0,0,1,1,1,1],
    [1,1,1,1,0,0,0,0]
    ]
# load label data, which must be a string array
y = [
      "cold",
      "cold",
      "hot",
      "hot"
    ]
addressSize = 3     # number of addressing bits in the ram
ignoreZero  = False # optional; causes the rams to ignore the address 0

# False by default for performance reasons,
# when True, WiSARD prints the progress of train() and classify()
verbose = True

#wsd = wp.Wisard(addressSize, ignoreZero=ignoreZero, verbose=verbose)

wsd = wp.Wisard(
  5, # addressSize
  bleachingActivated=True,
  ignoreZero=False,
  completeAddressing=True,
  verbose=True,
  indexes=[],
  base=256,
  confidence=1,
  returnActivationDegree=False,
  returnConfidence=False,
  returnClassesDegrees=False
)
# train using the input data
wsd.train(X, y)

# classify some data
out = wsd.classify([X[1]])
print(out)
exit(0)
# the output of classify is a string list in the same sequence as the input
for i, d in enumerate(X):
    print(out[i], d)

