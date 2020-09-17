package algnum

import (
	"math/rand"
	"testing"
	"time"
)

func TestScalarProd(t *testing.T) {
	a := []float64{1, 2, 3}
	b := []float64{4, 5, 6}

	res, err := ScalarProd(a, b)

	if err != nil {
		t.Fatal(err)
	} else if res != 32 {
		t.Fatalf("result is wrong: expected %f, got %f", 32.0, res)
	} else {
		t.Log("scalar product works correct")
	}
}

func TestMatVecMul(t *testing.T) {
	data := [][]float64{
		{1, 2, 3},
		{4, 5, 6},
		{7, 8, 9},
	}

	mat, _ := InitMat(data)

	x := []float64{1, 2, 3}

	expectedRes := []float64{14, 32, 50}
	res, err := MatVecMul(mat, x)

	if err != nil {
		t.Fatal(err)
	} else if !VectsEq(res, expectedRes) {
		t.Fatalf("result is wrong: expected %s, got %s", VectToStr(expectedRes), VectToStr(res))
	} else {
		t.Log("matrix to vector multiplication works correct")
	}
}

func TestMatMul(t *testing.T) {
	aData := [][]float64{
		{1, 2, 3},
		{4, 5, 6},
		{7, 8, 9},
	}
	a, _ := InitMat(aData)

	bData := [][]float64{
		{1, 2, 3},
		{4, 5, 6},
		{7, 8, 9},
	}
	b, _ := InitMat(bData)

	expectedResData := [][]float64{
		{30, 36, 42},
		{66, 81, 96},
		{102, 126, 150},
	}
	expectedRes, _ := InitMat(expectedResData)

	res, err := MatMul(a, b)
	if err != nil {
		t.Fatal(err)
	} else if !MatsEq(res, expectedRes) {
		t.Fatalf("result is wrong: expected\n %s,\ngot\n %s", expectedRes.ToStr(), res.ToStr())
	} else {
		t.Log("matrixs multiplication works correct")
	}
}

func TestSwitches(t *testing.T) {
	data := [][]float64{
		{1, 2, 3},
		{4, 5, 6},
		{7, 8, 9},
	}
	mat, _ := InitMat(data)

	expSwitchRowsData := [][]float64{
		{4, 5, 6},
		{1, 2, 3},
		{7, 8, 9},
	}
	expSwitchRows, _ := InitMat(expSwitchRowsData)

	expSwitchColsData := [][]float64{
		{2, 1, 3},
		{5, 4, 6},
		{8, 7, 9},
	}
	expSwitchCols, _ := InitMat(expSwitchColsData)

	switchRows, err := SwitchRows(mat, 0, 1)
	if err != nil {
		t.Fatal(err)
	}

	switchCols, err := SwitchCols(mat, 0, 1)
	if err != nil {
		t.Fatal(err)
	}

	if !MatsEq(expSwitchRows, switchRows) {
		t.Fatalf("switch rows is wrong: expected\n %s,\ngot\n %s", expSwitchRows.ToStr(), switchRows.ToStr())
	}
	if !MatsEq(expSwitchCols, switchCols) {
		t.Fatalf("switch cols is wrong: expected\n %s,\ngot\n %s", expSwitchCols.ToStr(), switchCols.ToStr())
	}

	t.Log("switches work correct")
}

func randData(rows, cols int, max int) [][]float64 {
	rand.Seed(time.Now().UnixNano())

	res := make([][]float64, rows)
	for i := range res {
		res[i] = make([]float64, cols)
	}

	for i := range res {
		for j := range res {
			res[i][j] = float64(rand.Intn(max) + 1)
		}
	}

	return res
}

func randFree(dim int, max int) []float64 {
	rand.Seed(time.Now().UnixNano())

	res := make([]float64, dim)
	for i := range res {
		res[i] = float64(rand.Intn(max) + 1)
	}

	return res
}

func TestGauss(t *testing.T) {
	data := [][]float64{
		{0, 2, 3},
		{4, 5, 6},
		{7, 8, 9},
	}
	mat, _ := InitMat(data)
	f := []float64{1, 2, 3}

	expectedRes := []float64{0, 0, float64(1)/float64(3)}

	res, err := Gauss(mat, f)
	if err != nil {
		t.Fatal(err)
	} else if !VectsEq(expectedRes, res) {
		t.Fatalf("result is wrong: expected\n %s,\ngot\n %s", VectToStr(expectedRes), VectToStr(res))
	} else {
		t.Log("gauss works correct, input:\nA = ", mat.ToStr(), "\nf = ", VectToStr(f), "\nresult:", VectToStr(res))
	}

	data2 := [][]float64{
		{0, 2, 3},
		{2, 0, 3},
		{0, 5, 0},
	}
	mat2, _ := InitMat(data2)
	f2 := []float64{4, 7, 1}

	expectedRes2 := []float64{float64(17)/float64(10), float64(1)/float64(5), float64(6)/float64(5)}

	res2, err := Gauss(mat2, f2)
	if err != nil {
		t.Fatal(err)
	} else if !VectsEq(expectedRes2, res2) {
		t.Fatalf("result is wrong: expected\n %s,\ngot\n %s", VectToStr(expectedRes2), VectToStr(res2))
	} else {
		t.Log("gauss works correct, input:\nA = ", mat2.ToStr(), "\nf = ", VectToStr(f2), "\nresult:", VectToStr(res2))
	}

	data3 := [][]float64{
		{4,	0, 6, 8, 8,	3},
		{15, 43, 0,	2, 2, 0},
		{0, 54, 13,	13,	5, 87},
		{14, 15, 34, 32, 0,	0},
		{11, 0,	0, 14, 0, 46},
		{43, 23, 0,	0, 0, 0},
	}
	mat3, _ := InitMat(data3)
	f3 := []float64{35, 44, 6, 24, 76, 2}

	expectedRes3 := []float64{-float64(14957666)/float64(47372961), float64(32083720)/float64(47372961), -float64(318455426)/float64(47372961), float64(121797782)/float64(15790987), float64(99194311)/float64(47372961), -float64(29361467)/float64(47372961)}

	res3, err := Gauss(mat3, f3)
	if err != nil {
		t.Fatal(err)
	} else if !VectsEq(expectedRes3, res3) {
		t.Fatalf("result is wrong: expected\n %s,\ngot\n %s", VectToStr(expectedRes3), VectToStr(res3))
	} else {
		t.Log("gauss works correct, input:\nA = ", mat3.ToStr(), "\nf = ", VectToStr(f3), "\nresult:", VectToStr(res3))
	}

	data4 := [][]float64{
		{0, 2, 3},
		{0, 0, 0},
		{1, 0, 0},
	}
	mat4, _ := InitMat(data4)
	f4 := []float64{2, 3, 0}

	res4, err := Gauss(mat4, f4)
	if err != nil {
		t.Log("gauss works correct, got expected error:", err)
	} else {
		t.Log("result is wrong, input:\nA = ", mat4.ToStr(), "\nf = ", VectToStr(f4), "\nexpected error, got:\n", VectToStr(res4))
	}

	dataN := randData(100, 100, 1000)
	matN, _ := InitMat(dataN)
	fN := randFree(100, 1000)

	resN, err := Gauss(matN, fN)
	if err != nil {
		t.Fatal(err)
	}
	check, err := MatVecMul(matN, resN)
	if err != nil {
		t.Fatal(err)
	} else if !VectsEq(fN, check) {
		t.Fatalf("result is wrong, input:\nA = %s\nf = %s\nres = %s\ncheck:%s", matN.ToStr(), VectToStr(fN), VectToStr(resN), VectToStr(check))
	} else {
		t.Log("gauss works correct, input:\nA = ", matN.ToStr(), "\nf = ", VectToStr(fN), "\nresult:", VectToStr(resN), "\ncheck:", VectToStr(check))
	}

	data5 := [][]float64{
		{-0.0000001, 1},
		{1, 2},
	}
	mat5, _ := InitMat(data5)
	f5 := []float64{1, 4}

	expectedRes5 := []float64{2, 1}

	res5, err := Gauss(mat5, f5)
	if err != nil {
		t.Fatal(err)
	} else if !VectsEq(expectedRes5, res5) {
		t.Fatalf("result is wrong: expected\n %s,\ngot\n %s", VectToStr(expectedRes5), VectToStr(res5))
	} else {
		t.Log("gauss works correct, input:\nA = ", mat5.ToStr(), "\nf = ", VectToStr(f5), "\nresult:", VectToStr(res5))
	}
}