package algnum

import (
	"errors"
	"fmt"
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

func copy2dSlice(src [][]float64) [][]float64 {
	dst := make([][]float64, len(src))
	for i := range dst {
		dst[i] = make([]float64, len(src[i]))
		copy(dst[i], src[i])
	}

	return dst
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

	data := make([][]float64, dim)
	for i := range data {
		data[i] = make([]float64, dim)
	}

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

	data := make([][]float64, rows)
	for i := range data {
		data[i] = make([]float64, cols)
	}

	return &Matrix{data: data, rows: rows, cols: cols}, nil
}

func VectsEq(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}

	for i, ai := range a {
		if ai != b[i] {
			return false
		}
	}

	return true
}

func MatsEq(a, b *Matrix) bool {
	if a.rows != b.rows || a.cols != b.cols {
		return false
	}

	for i, ai := range a.data {
		if !VectsEq(ai, b.data[i]) {
			return false
		}
	}

	return true
}

func VectToStr(a []float64) string {
	res := "["
	for _, ai := range a {
		res += fmt.Sprintf("%f", ai) + ", "
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

	res := a

	remember := res[i]
	res[i] = res[j]
	res[j] = remember

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

	remember := resData[i]
	resData[i] = resData[j]
	resData[j] = remember

	return &Matrix{data: resData, rows: mat.rows, cols: mat.cols}, nil
}

func SwitchCols(mat *Matrix, i, j int) (*Matrix, error) {
	if i >= mat.cols || j >= mat.cols || i < 0 || j < 0 {
		return nil, errors.New("wrong indexes")
	}

	resData := copy2dSlice(mat.data)

	for k := range resData {
		remember := resData[k][i]
		resData[k][i] = resData[k][j]
		resData[k][j] = remember
	}

	return &Matrix{data: resData, rows: mat.rows, cols: mat.cols}, nil
}

func forwElim(a *Matrix, f []float64) (*Matrix, []float64, error) {
	if len(a.data) == 0 || len(f) == 0 {
		return nil, nil, errors.New("matrix or free element is empty")
	}

	resMat := &Matrix{data: copy2dSlice(a.data), rows: a.rows, cols: a.cols}
	resF := f

	for i := range resMat.data {
		leadEl := resMat.data[i][i]
		if leadEl == 0 {
			for j := i; j < resMat.rows; j++ {
				if leadEl = resMat.data[j][i]; leadEl != 0 {
					if j == 0 {
						break
					}
					resMat, _ = SwitchRows(resMat, 0, j)
					resF, _ = SwitchVecEls(resF, 0, j)
					break
				}
			}
		}

		if leadEl == 0 {
			if resF[i] == 0 {
				return nil, nil, errors.New("system has infinite solutions")
			} else {
				return nil, nil, errors.New("system has no solutions")
			}
		}

		resMat.data[i][i] = 1
		for j := i + 1; j < resMat.rows; j++ {
			resMat.data[i][j] /= leadEl
		}
		f[i] /= leadEl

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