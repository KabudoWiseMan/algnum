package algnum

import (
	"errors"
	"fmt"
	"math"
	"sync"
)

const (
	EuclideanNorm = iota
	InfinityNorm
	Epsilon = 1e-6
)

type Matrix struct {
	data [][]float64
	rows, cols int
}

func (mat *Matrix) Rows() int {
	return mat.rows
}

func (mat *Matrix) Cols() int {
	return mat.cols
}

func (mat *Matrix) IsSquare() bool {
	return mat.rows == mat.cols
}

func InitMat(data [][]float64) (*Matrix, error) {
	if len(data) == 0 {
		return &Matrix{}, nil
	}

	rows := len(data)
	cols := len(data[0])
	for _, d := range data {
		if len(d) != cols {
			return nil, errors.New("could't create matrix")
		}
	}

	return &Matrix{data: copy2dSlice(data), rows: rows, cols: cols}, nil
}

func InitMatOfDim(dim int) (*Matrix, error) {
	if dim < 0 {
		return nil, errors.New("wrong dimention")
	}
	if dim == 0 {
		return &Matrix{}, nil
	}

	data := init2dSlice(dim, dim)

	return &Matrix{data: data, rows: len(data), cols: len(data[0])}, nil
}

func InitMatOfDims(rows, cols int) (*Matrix, error) {
	if rows < 0 || cols < 0 {
		return nil, errors.New("wrong dimentions")
	}
	if rows == 0 || cols == 0 {
		if rows == cols {
			return &Matrix{}, nil
		} else {
			return nil, errors.New("wrong dimentions")
		}
	}

	data := init2dSlice(rows, cols)

	return &Matrix{data: data, rows: rows, cols: cols}, nil
}

func (m *Matrix) IsDiagDominant() bool {
	if !m.IsSquare() {
		return false
	}
	for i := 0; i < m.rows; i++ {
		aii := math.Abs(m.data[i][i])
		var sum float64
		for j := 0; j < m.cols; j++ {
			if j == i {
				continue
			}
			sum += math.Abs(m.data[i][j])
			if aii < sum {
				return false
			}
		}
	}
	return true
}

func (m *Matrix) Det() (float64, error) {
	if !m.IsSquare() {
		return -1, errors.New("matrix isn't square")
	}

	return detRec(m.data), nil
}

func del(data [][]float64, row, col int) [][]float64 {
	res := init2dSlice(len(data) - 1, len(data[0]) - 1)
	var k, l int
	for i := range data {
		if i == row {
			continue
		}
		l = 0
		for j := range data[i] {
			if j == col {
				continue
			}
			res[k][l] = data[i][j]
			l++
		}
		k++
	}
	return res
}

func detRec(data [][]float64) float64 {
	dim := len(data)
	if dim == 1 {
		return data[0][0]
	}
	if dim == 2 {
		return data[0][0] * data[1][1] - data[0][1] * data[1][0]
	}

	res := float64(0)
	sign := float64(1)
	for i := 0; i < dim; i++ {
		res += sign * data[i][0] * detRec(del(data, i, 0))
		sign *= -1
	}

	return res
}

func adjoint(data [][]float64) [][]float64 {
	if len(data) == 1 {
		return [][]float64{{data[0][0]}}
	}

	res := init2dSlice(len(data), len(data[0]))
	sign := float64(-1)

	var wg sync.WaitGroup
	for i := range data {
		for j := range data[i] {
			wg.Add(1)
			go func(res [][]float64, i, j int, sign *float64) {
				defer wg.Done()
				cofactor := del(data, i, j)
				if (i + j) % 2 == 0 {
					*sign = 1
				} else {
					*sign = -1
				}

				res[j][i] = *sign * detRec(cofactor)
			}(res, i, j, &sign)
		}
	}

	wg.Wait()
	return res
}

func (m *Matrix) Inverse() (*Matrix, error) {
	det, err := m.Det()
	if err != nil {
		return nil, err
	}
	if det == 0 {
		return nil, errors.New("matrix is singular")
	}

	invData := init2dSlice(m.rows, m.cols)

	adj := adjoint(m.data)
	var wg sync.WaitGroup
	for i := range m.data {
		for j := range m.data[i] {
			wg.Add(1)
			go func(invData [][]float64, i, j int) {
				defer wg.Done()
				invData[i][j] = adj[i][j] / det
			}(invData, i, j)
		}
	}

	wg.Wait()
	inv := &Matrix{invData, m.rows, m.cols}
	return inv, nil
}

func (m *Matrix) Norm(normType int) float64 {
	var norm float64

	switch normType {
	case InfinityNorm:
		for i := 0; i < m.rows; i++ {
			var absRowSum float64
			for j := 0; j < m.cols; j++ {
				absRowSum += math.Abs(m.data[i][j])
			}
			if absRowSum >= norm {
				norm = absRowSum
			}
		}
		return norm
	default:
		for i := 0; i < m.rows; i++ {
			for j := 0; j < m.cols; j++ {
				norm += m.data[i][j] * m.data[i][j]
			}
		}
		return math.Sqrt(norm)
	}
}

func VectsEq(a, b []float64, eps float64) bool {
	if len(a) != len(b) {
		return false
	}

	for i, ai := range a {
		if math.Abs(ai - b[i]) >= eps {
			return false
		}
	}

	return true
}

func MatsSlice(m *Matrix, i, n, j, k int) (*Matrix, error) {
	if i < 0 || j < 0 || n > m.rows || k > m.cols {
		return nil, errors.New("indexes are out of bounds")
	}

	return &Matrix{copy2dSlice(m.data[i : n][j : k]), n - i, k - j}, nil
}

func MatsSum(a, b *Matrix) (*Matrix, error) {
	if a.rows != b.rows || a.cols != b.cols {
		return nil, errors.New("matrixs dims don't match")
	}

	resData := copy2dSlice(a.data)
	for i := range resData {
		for j := range resData[i] {
			resData[i][j] += b.data[i][j]
		}
	}
	res := &Matrix{resData, a.rows, a.cols}

	return res, nil
}

func MatsSub(a, b *Matrix) (*Matrix, error) {
	if a.rows != b.rows || a.cols != b.cols {
		return nil, errors.New("matrixs dims don't match")
	}

	resData := copy2dSlice(a.data)
	for i := range resData {
		for j := range resData[i] {
			resData[i][j] -= b.data[i][j]
		}
	}
	res := &Matrix{resData, a.rows, a.cols}

	return res, nil
}

func slicesSum(a, b [][]float64) [][]float64 {
	res := copy2dSlice(a)
	for i := range res {
		for j := range res[i] {
			res[i][j] += b[i][j]
		}
	}
	return res
}

func slicesSub(a, b [][]float64) [][]float64 {
	res := copy2dSlice(a)
	for i := range res {
		for j := range res[i] {
			res[i][j] -= b[i][j]
		}
	}
	return res
}

func MatsEq(a, b *Matrix, eps float64) bool {
	if a.rows != b.rows || a.cols != b.cols {
		return false
	}

	for i, ai := range a.data {
		if !VectsEq(ai, b.data[i], eps) {
			return false
		}
	}

	return true
}

func VectToStr(a []float64) string {
	res := "["
	for _, ai := range a {
		res += fmt.Sprintf("%0.15f", ai) + ", "
	}
	if len(res) > 1 {
		res = res[:len(res) - 2]
	}

	res += "]"

	return res
}

func (mat *Matrix) ToStr() string {
	res := "["

	for _, m := range mat.data {
		res += VectToStr(m) + ",\n"
	}

	if len(res) > 1 {
		res = res[:len(res) - 2]
	}

	res += "]"

	return res
}

func VecNorm(x []float64, normType int) float64 {
	var norm float64

	switch normType {
	case InfinityNorm:
		for _, xi := range x {
			xiAbs := math.Abs(xi)
			if xiAbs >= norm {
				norm = xiAbs
			}
		}
		return norm
	default:
		for _, xi := range x {
			norm += xi * xi
		}
		return math.Sqrt(norm)
	}
}

func ScalarProd(a, b []float64) (float64, error) {
	if len(a) != len(b) {
		return -1, errors.New("vector dims don't match")
	}

	if len(a) == 0 {
		return -1, errors.New("vectors are empty")
	}

	res := float64(0)
	for i, ai := range a {
		res += ai * b[i]
	}

	return res, nil
}

func VecsSum(a, b []float64) ([]float64, error) {
	if len(a) != len(b) {
		return nil, errors.New("vector dims don't match")
	}

	res := make([]float64, len(a))
	for i, ai := range a {
		res[i] = ai + b[i]
	}

	return res, nil
}

func VecsSub(a, b []float64) ([]float64, error) {
	if len(a) != len(b) {
		return nil, errors.New("vector dims don't match")
	}

	res := make([]float64, len(a))
	for i, ai := range a {
		res[i] = ai - b[i]
	}

	return res, nil
}

func VecsMul(a, b []float64) ([]float64, error) {
	if len(a) != len(b) {
		return nil, errors.New("vector dims don't match")
	}

	res := make([]float64, len(a))
	for i, ai := range a {
		res[i] = ai * b[i]
	}

	return res, nil
}

func VecsDiv(a, b []float64) ([]float64, error) {
	if len(a) != len(b) {
		return nil, errors.New("vector dims don't match")
	}

	res := make([]float64, len(a))
	for i, ai := range a {
		res[i] = ai / b[i]
	}

	return res, nil
}

func VecConstMul(a []float64, c float64) []float64 {
	res := make([]float64, len(a))
	for i, ai := range a {
		res[i] = ai * c
	}

	return res
}

func SwitchVecEls(a []float64, i, j int) ([]float64, error) {
	if i >= len(a) || j >= len(a) || i < 0 || j < 0 {
		return nil, errors.New("wrong indexes")
	}

	res := make([]float64, len(a))
	copy(res, a)

	res[i], res[j] = res[j], res[i]

	return res, nil
}

func MatVecMul(a *Matrix, x []float64) ([]float64, error) {
	if a.rows == 0 {
		return nil, errors.New("matrix is empty")
	}

	if a.rows != len(x) {
		return nil, errors.New("matrix and vector dims don't match")
	}

	res := make([]float64, len(x))
	for i, ai := range a.data {
		scalProd, _ := ScalarProd(ai, x)
		res[i] = scalProd
	}

	return res, nil
}

func TransposeMat(mat *Matrix) *Matrix {
	transMat, _ := InitMatOfDims(mat.rows, mat.cols)

	for i := range mat.data {
		for j := range mat.data {
			transMat.data[j][i] = mat.data[i][j]
		}
	}

	return transMat
}

func MatMul(a, b *Matrix) (*Matrix, error) {
	if a.cols != b.rows {
		return nil, errors.New("matrix dims don't match")
	}

	transRes, _ := InitMatOfDims(b.rows, b.cols)

	transB := TransposeMat(b)
	for i, bi := range transB.data {
		row, _ := MatVecMul(a, bi)
		transRes.data[i] = row
	}

	res := TransposeMat(transRes)

	return res, nil
}

func SwitchRows(mat *Matrix, i, j int) (*Matrix, error) {
	if i >= mat.rows || j >= mat.rows || i < 0 || j < 0 {
		return nil, errors.New("wrong indexes")
	}

	resData := copy2dSlice(mat.data)

	resData[i], resData[j] = resData[j], resData[i]

	return &Matrix{data: resData, rows: mat.rows, cols: mat.cols}, nil
}

func SwitchCols(mat *Matrix, i, j int) (*Matrix, error) {
	if i >= mat.cols || j >= mat.cols || i < 0 || j < 0 {
		return nil, errors.New("wrong indexes")
	}

	resData := copy2dSlice(mat.data)

	for k := range resData {
		resData[k][i], resData[k][j] = resData[k][j], resData[k][i]
	}

	return &Matrix{data: resData, rows: mat.rows, cols: mat.cols}, nil
}