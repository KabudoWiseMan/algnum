package algnum

import (
	"errors"
	"fmt"
)

type Matrix struct {
	data [][]float64
	dim int
}

func InitMat(data [][]float64) (*Matrix, error) {
	if len(data) == 0 {
		return &Matrix{}, nil
	}

	dim := len(data)
	for _, d := range data {
		if len(d) != dim {
			return nil, errors.New("matrix isn't square")
		}
	}

	return &Matrix{data: data, dim: dim}, nil
}

func InitMatOfDim(dim int) *Matrix {
	data := make([][]float64, dim)
	for i := range data {
		data[i] = make([]float64, dim)
	}
	mat, _ := InitMat(data)
	return mat
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
	if a.dim != b.dim {
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

func (mat *Matrix) toStr() string {
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
		bi := b[i]
		res += ai * bi
	}

	return res, nil
}

func MatVecMul(a *Matrix, x []float64) ([]float64, error) {
	if a.dim == 0 {
		return nil, errors.New("matrix is empty")
	}

	if a.dim != len(x) {
		return nil, errors.New("matrix and vector dims don't match")
	}

	var res []float64
	for _, ai := range a.data {
		scalProd, _ := ScalarProd(ai, x)
		res = append(res, scalProd)
	}

	return res, nil
}

func TransposeMat(mat *Matrix) *Matrix {
	transMat := InitMatOfDim(mat.dim)

	for i, _ := range mat.data {
		for j, _ := range mat.data {
			transMat.data[j][i] = mat.data[i][j]
		}
	}

	return transMat
}

func MatMul(a, b *Matrix) (*Matrix, error) {
	if a.dim != b.dim {
		return nil, errors.New("matrix dims don't match")
	}

	transRes := InitMatOfDim(a.dim)

	transB := TransposeMat(b)
	for i, bi := range transB.data {
		row, _ := MatVecMul(a, bi)
		transRes.data[i] = row
	}

	res := TransposeMat(transRes)

	return res, nil
}