package algnum

import (
	"errors"
	"math"
	"sync"
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

	if a.IsDiagDominant() {
		diagMat, newF, err := forwElim(a, f)
		if err != nil {
			return nil, err
		}

		res := backSubs(diagMat, newF)

		return res, nil
	} else {
		if !a.IsSquare() {
			return nil, errors.New("matrix isn't square")
		}
		diagMat, newF, idxs, err := forwElimLeadEl(a, f)
		if err != nil {
			return nil, err
		}

		res := backSubsLeadEl(diagMat, newF, idxs)

		return res, nil
	}
}

func GaussClassic(a *Matrix, f []float64) ([]float64, error) {
	if a.rows != len(f) {
		return nil, errors.New("matrix and free element dims don't match")
	}
	if a.rows == 0 || len(f) == 0 {
		return nil, errors.New("matrix or free element is empty")
	}
	if !a.IsSquare() {
		return nil, errors.New("matrix isn't square")
	}

	diagMat, newF, err := forwElim(a, f)
	if err != nil {
		return nil, err
	}

	res := backSubs(diagMat, newF)

	return res, nil
}

func GaussWithLeadEl(a *Matrix, f []float64) ([]float64, error) {
	if a.rows != len(f) {
		return nil, errors.New("matrix and free element dims don't match")
	}
	if a.rows == 0 || len(f) == 0 {
		return nil, errors.New("matrix or free element is empty")
	}
	if !a.IsSquare() {
		return nil, errors.New("matrix isn't square")
	}
	
	diagMat, newF, idxs, err := forwElimLeadEl(a, f)
	if err != nil {
		return nil, err
	}

	res := backSubsLeadEl(diagMat, newF, idxs)

	return res, nil
}

func normVec(v []float64) float64 {
	max := float64(0)
	for _, val := range v {
		max = math.Max(max, math.Abs(val))
	}

	return max
}

func Jacobi(a *Matrix, f []float64) ([]float64, error) {
	n := len(f)
	if a.rows != n {
		return nil, errors.New("matrix and free element dims don't match")
	}
	if a.rows == 0 || n == 0 {
		return nil, errors.New("matrix or free element is empty")
	}
	if !a.IsDiagDominant() {
		return nil, errors.New("matrix isn't diagonally dominant")
	}

	x, xPrev := make([]float64, n), make([]float64, n)

	//iter := 0
	for {
		//iter++
		for i := 0; i < n; i++ {
			var sum float64
			for j := 0; j < n; j++ {
				if j != i {
					sum += a.data[i][j] * xPrev[j]
				}
			}
			x[i] = (f[i] - sum) / a.data[i][i]
		}
		sub, _ := VecsSub(x, xPrev)
		if VecNorm(sub, InfinityNorm) <= Epsilon {
			//fmt.Println("iterations:", iter)
			return x, nil
		}
		copy(xPrev, x)
	}
}

func Seidel(a *Matrix, f []float64) ([]float64, error) {
	n := len(f)
	if a.rows != n {
		return nil, errors.New("matrix and free element dims don't match")
	}
	if a.rows == 0 || n == 0 {
		return nil, errors.New("matrix or free element is empty")
	}
	if !a.IsDiagDominant() {
		return nil, errors.New("matrix isn't diagonally dominant")
	}

	x, xPrev := make([]float64, n), make([]float64, n)

	//iter := 0
	for {
		//iter++
		for i := 0; i < n; i++ {
			var sum float64
			for j := 0; j < n; j++ {
				if j != i {
					sum += a.data[i][j] * x[j]
				}
			}
			x[i] = (f[i] - sum) / a.data[i][i]
		}
		sub, _ := VecsSub(x, xPrev)
		if VecNorm(sub, InfinityNorm) <= Epsilon {
			//fmt.Println("iterations:", iter)
			return x, nil
		}
		copy(xPrev, x)
	}
}

func Strassen(a, b *Matrix, parallel bool) (*Matrix, error) {
	if a.cols != b.rows {
		return nil, errors.New("matrix dims don't match")
	}

	adj := false
	aData, bData := copy2dSlice(a.data), copy2dSlice(b.data)
	if !a.IsSquare() || !b.IsSquare() || !isPowerOfTwo(a.rows) {
		aData, bData = complementMatrixes(aData, bData, a.cols, b.cols)
		adj = true
	}

	var resData [][]float64
	if parallel {
		var wg sync.WaitGroup
		wg.Add(1)
		resData = strassRecPar(aData, bData, 64, &wg)
	} else {
		resData = strassRec(aData, bData, 64)
	}

	if adj {
		resData = adjustMatrix(resData, a.rows, b.cols)
	}

	return &Matrix{resData, a.rows, a.cols}, nil
}

func complementMatrixes(a, b [][]float64, aC, bC int) ([][]float64, [][]float64) {
	aR := len(a)
	bR := len(b)
	n := nearestPowerOfTwo(intMax4(aR, aC, bR, bC))

	aNew := init2dSlice(n, n)
	bNew := init2dSlice(n, n)
	for i := 0; i < aR; i++ {
		copy(aNew[i], a[i])
	}
	for i := 0; i < bR; i++ {
		copy(bNew[i], b[i])
	}

	return aNew, bNew
}

func adjustMatrix(m [][]float64, rows, cols int) [][]float64 {
	m = m[ : rows]
	for i := 0; i < rows; i++ {
		m[i] = m[i][ : cols]
	}
	return m
}

func divideSlice(a [][]float64, m int) ([][]float64, [][]float64, [][]float64, [][]float64) {
	a11 := copy2dSlice(a[ : m])
	a12 := copy2dSlice(a[ : m])
	a21 := copy2dSlice(a[m : ])
	a22 := copy2dSlice(a[m : ])
	for i := 0; i < m; i++ {
		a11[i] = a11[i][ : m]
		a12[i] = a12[i][m : ]
		a21[i] = a21[i][ : m]
		a22[i] = a22[i][m : ]
	}
	return a11, a12, a21, a22
}

func buildSlice(a11, a12, a21, a22 [][]float64) [][]float64 {
	m := len(a11)
	n := m * 2
	a := init2dSlice(n, n)

	for i := 0; i < m; i ++ {
		for j := 0; j < m; j++ {
			a[i][j] = a11[i][j]
		}
	}
	for i := 0; i < m; i ++ {
		for j := 0; j < m; j++ {
			a[i][m + j] = a12[i][j]
		}
	}
	for i := 0; i < m; i ++ {
		for j := 0; j < m; j++ {
			a[m + i][j] = a21[i][j]
		}
	}
	for i := 0; i < m; i ++ {
		for j := 0; j < m; j++ {
			a[m + i][m + j] = a22[i][j]
		}
	}

	return a
}

func strassRec(a, b [][]float64, nMin int) [][]float64 {
	n := len(a)
	if n <= nMin {
		c, _ := MatMul(&Matrix{a, n, n}, &Matrix{b, n, n})
		return c.data
	}

	m := n / 2

	a11, a12, a21, a22 := divideSlice(a, m)
	b11, b12, b21, b22 := divideSlice(b, m)

	p1 := strassRec(slicesSum(a11, a22), slicesSum(b11, b22), nMin)
	p2 := strassRec(slicesSum(a21, a22), b11, nMin)
	p3 := strassRec(a11, slicesSub(b12, b22), nMin)
	p4 := strassRec(a22, slicesSub(b21, b11), nMin)
	p5 := strassRec(slicesSum(a11, a12), b22, nMin)
	p6 := strassRec(slicesSub(a21, a11), slicesSum(b11, b12), nMin)
	p7 := strassRec(slicesSub(a12, a22), slicesSum(b21, b22), nMin)

	c11 := slicesSum(slicesSub(slicesSum(p1, p4), p5), p7)
	c12 := slicesSum(p3, p5)
	c21 := slicesSum(p2, p4)
	c22 := slicesSum(slicesSub(slicesSum(p1, p3), p2), p6)
	c := buildSlice(c11, c12, c21, c22)

	return c
}

func strassRecPar(a, b [][]float64, nMin int, wg *sync.WaitGroup) [][]float64 {
	defer wg.Done()
	n := len(a)
	if n <= nMin {
		c, _ := MatMul(&Matrix{a, n, n}, &Matrix{b, n, n})
		return c.data
	}

	m := n / 2

	a11, a12, a21, a22 := divideSlice(a, m)
	b11, b12, b21, b22 := divideSlice(b, m)

	var wg2 sync.WaitGroup
	wg2.Add(7)
	var p1, p2, p3, p4, p5, p6, p7 [][]float64
	go func() { p1 = strassRecPar(slicesSum(a11, a22), slicesSum(b11, b22), nMin, &wg2) }()
	go func() { p2 = strassRecPar(slicesSum(a21, a22), b11, nMin, &wg2) }()
	go func() { p3 = strassRecPar(a11, slicesSub(b12, b22), nMin, &wg2) }()
	go func() { p4 = strassRecPar(a22, slicesSub(b21, b11), nMin, &wg2) }()
	go func() { p5 = strassRecPar(slicesSum(a11, a12), b22, nMin, &wg2) }()
	go func() { p6 = strassRecPar(slicesSub(a21, a11), slicesSum(b11, b12), nMin, &wg2) }()
	go func() { p7 = strassRecPar(slicesSub(a12, a22), slicesSum(b21, b22), nMin, &wg2) }()
	wg2.Wait()

	c11 := slicesSum(slicesSub(slicesSum(p1, p4), p5), p7)
	c12 := slicesSum(p3, p5)
	c21 := slicesSum(p2, p4)
	c22 := slicesSum(slicesSub(slicesSum(p1, p3), p2), p6)
	c := buildSlice(c11, c12, c21, c22)

	return c
}

func Cholesky(a *Matrix) (*Matrix, error) {
	if !a.IsDiagDominant() {
		return nil, errors.New("matrix isn't diagonally dominant")
	}

	n := a.rows
	resData := init2dSlice(n, n)

	for i := 0; i < n; i++ {
		s := a.data[i][i]
		for ip := 0; ip < i - 1; ip++ {
			s = s - resData[i][ip] * resData[i][ip]
		}
		resData[i][i] = math.Sqrt(s)
		for j := i + 1; j < n; j++ {
			s = a.data[j][i]
			for ip := 0; ip < i - 1; ip++ {
				s = s - resData[i][ip] * resData[j][ip]
			}
			resData[j][i] = s / resData[i][i]
		}
	}

	return &Matrix{resData, n, n}, nil
}