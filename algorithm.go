package algnum

import (
	"errors"
	"math"
)

func forwElim(a *Matrix, f []float64) (*Matrix, []float64, error) {
	resMat := &Matrix{data: copy2dSlice(a.data), rows: a.rows, cols: a.cols}
	resF := make([]float64, len(f))
	copy(resF, f)

	for i := range resMat.data {
		leadEl := resMat.data[i][i]

		resMat.data[i][i] = 1
		for j := i + 1; j < resMat.rows; j++ {
			resMat.data[i][j] /= leadEl
		}
		resF[i] /= leadEl

		for j := i + 1; j < resMat.rows; j++ {
			el := resMat.data[j][i]
			resMat.data[j][i] = 0
			for k := i + 1; k < resMat.cols; k++ {
				resMat.data[j][k] -= resMat.data[i][k] * el
			}
			resF[j] -= resF[i] * el
		}
	}

	return resMat, resF, nil
}

func backSubs(a *Matrix, f []float64) []float64 {
	x := make([]float64, len(f))

	x[len(x) - 1] = f[len(f) - 1]

	for i := a.rows - 2; i >= 0; i-- {
		x[i] = f[i]
		for j := a.cols - 1; j > i; j-- {
			x[i] -= a.data[i][j] * x[j]
		}
	}

	return x
}

func forwElimLeadEl(a *Matrix, f []float64) (*Matrix, []float64, []int, error) {
	n := a.rows

	resMat := &Matrix{data: copy2dSlice(a.data), rows: n, cols: n}
	resF := make([]float64, n)
	copy(resF, f)

	idxs := make([]int, n)
	for i := range idxs {
		idxs[i] = i
	}

	for i := 0; i < n; i++ {
		// with leading col element
		leadColElMod := math.Abs(resMat.data[i][i])

		k := i
		for j := i + 1; j < n; j++ {
			if math.Abs(resMat.data[j][i]) > leadColElMod {
				leadColElMod = math.Abs(resMat.data[j][i])
				k = j
			}
		}

		if leadColElMod == 0 {
			return nil, nil, nil, errors.New("system is not inconsistent")
		}

		if k != i {
			for j := i; j < n; j++ {
				resMat.data[i][j], resMat.data[k][j] = resMat.data[k][j], resMat.data[i][j]
			}
			resF[i], resF[k] = resF[k], resF[i]
		}

		leadEl := resMat.data[i][i]

		for j := i + 1; j < n; j++ {
			c := -resMat.data[j][i] / leadEl
			for l := i; l < n; l++ {
				resMat.data[j][l] += resMat.data[i][l] * c
			}
			resF[j] += resF[i] * c
		}

		// with leading row element
		leadRowElMod := math.Abs(resMat.data[i][i])

		k = i
		for j := i + 1; j < n; j++ {
			if math.Abs(resMat.data[i][j]) > leadRowElMod {
				leadRowElMod = math.Abs(resMat.data[i][j])
				k = j
			}
		}

		if leadRowElMod == 0 {
			return nil, nil, nil, errors.New("system is not inconsistent")
		}

		if k != i {
			for j := 0; j < n; j++ {
				resMat.data[j][i], resMat.data[j][k] = resMat.data[j][k], resMat.data[j][i]
			}
		}
		idxs[i], idxs[k] = idxs[k], idxs[i]

		leadEl = resMat.data[i][i]

		for j := i + 1; j < n; j++ {
			c := -resMat.data[j][i] / leadEl
			for l := i; l < n; l++ {
				resMat.data[j][l] += resMat.data[i][l] * c
			}
			resF[j] += resF[i] * c
		}
	}

	return resMat, resF, idxs, nil
}

func backSubsLeadEl(a *Matrix, f []float64, idxs []int) []float64 {
	n := a.rows

	x := make([]float64, n)

	x[idxs[n - 1]] = f[n - 1] / a.data[n - 1][n - 1]

	for i := n - 2; i >= 0; i-- {
		x[idxs[i]] = f[i]
		for j := n - 1; j > i; j-- {
			x[idxs[i]] -= a.data[i][j] * x[idxs[j]]
		}
		x[idxs[i]] /= a.data[i][i]
	}

	return x
}

func Gauss(a *Matrix, f []float64) ([]float64, error) {
	if a.rows != len(f) {
		return nil, errors.New("matrix and free element dims don't match")
	}
	if a.rows == 0 || len(f) == 0 {
		return nil, errors.New("matrix or free element is empty")
	}
	if !a.IsSquare() {
		return nil, errors.New("matrix isn't square")
	}

	if a.IsDiagDominant() {
		diagMat, newF, err := forwElim(a, f)
		if err != nil {
			return nil, err
		}

		res := backSubs(diagMat, newF)

		return res, nil
	} else {
		diagMat, newF, idxs, err := forwElimLeadEl(a, f)
		if err != nil {
			return nil, err
		}

		res := backSubsLeadEl(diagMat, newF, idxs)

		return res, nil
	}
}
