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
		t.Fatalf("result is wrong: expected\n %s,\ngot\n %s", expectedRes.toStr(), res.toStr())
	} else {
		t.Log("matrixs multiplication works correct")
	}
}