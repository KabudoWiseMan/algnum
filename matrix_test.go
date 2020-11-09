package algnum

import (
	"testing"
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
	} else if !VectsEq(res, expectedRes, Epsilon) {
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
	} else if !MatsEq(res, expectedRes, Epsilon) {
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

	if !MatsEq(expSwitchRows, switchRows, Epsilon) {
		t.Fatalf("switch rows is wrong: expected\n %s,\ngot\n %s", expSwitchRows.ToStr(), switchRows.ToStr())
	}
	if !MatsEq(expSwitchCols, switchCols, Epsilon) {
		t.Fatalf("switch cols is wrong: expected\n %s,\ngot\n %s", expSwitchCols.ToStr(), switchCols.ToStr())
	}

	t.Log("switches work correct")
}

func TestMatrix_Det(t *testing.T) {
	data := [][]float64{
		{3, 2, 1},
		{1, 3, 2},
		{1, 2, 4},
	}
	mat, _ := InitMat(data)

	expectedRes := float64(19)

	res, err := mat.Det()
	if err != nil {
		t.Fatal(err)
	} else if res != expectedRes {
		t.Fatalf("result is wrong: expected %0.15f, got %0.15f", expectedRes, res)
	} else {
		t.Log("det works correct")
	}
}

func TestMatrix_Inverse(t *testing.T) {
	data := [][]float64{
		{3, 2, 1},
		{1, 3, 2},
		{1, 2, 4},
	}
	mat, _ := InitMat(data)

	expectedRes := [][]float64{
		{float64(8) / float64(19), float64(-6) / float64(19), float64(1) / float64(19)},
		{float64(-2) / float64(19), float64(11) / float64(19), float64(-5) / float64(19)},
		{float64(-1) / float64(19), float64(-4) / float64(19), float64(7) / float64(19)},
	}
	exp, _ := InitMat(expectedRes)

	res, err := mat.Inverse()
	if err != nil {
		t.Fatal(err)
	} else if !MatsEq(exp, res, Epsilon) {
		t.Fatalf("result is wrong: expected\n %s,\ngot\n %s", exp.ToStr(), res.ToStr())
	} else {
		t.Log("inverse matrix works correct")
	}
}