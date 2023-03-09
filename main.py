
import wisardpkg as wp

X = [
      [1,1,1,0,0,0,0,0],
      [1,1,1,1,0,0,0,0],
      [0,0,0,0,1,1,1,1],
      [0,0,0,0,0,1,1,1]
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

wsd = wp.Wisard(addressSize, ignoreZero=ignoreZero, verbose=verbose)

# train using the input data
wsd.train(X, y)

# classify some data
out = wsd.classify(X2)

# the output of classify is a string list in the same sequence as the input
for i, d in enumerate(X2):
    print(out[i], d)


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
