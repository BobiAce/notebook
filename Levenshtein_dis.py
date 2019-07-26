import numpy as np

def Lev_distance():
    A = "fafasa"
    B = "fafsa"

    dp = np.array(np.arange(len(B)+1))

    for i in range(1, len(A)+1):
        temp1 = dp[0]
        dp[0] += 1
        for j in range(1, len(B)+1):
            temp2 = dp[j]
            if A[i-1] == B[j-1]:
                dp[j] = temp1
            else:
                dp[j] = min(temp1, min(dp[j-1], dp[j]))+1
            temp1 = temp2

    print("Levenshtein distance: {}".format(dp[len(B)]))


if __name__ == "__main__":
    Lev_distance()
