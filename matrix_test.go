package algnum

import (
	"gonum.org/v1/gonum/mat"
	"testing"
)

func lapackDet(m [][]float64, n int) float64 {
	var mVec []float64
	for i := 0; i < n; i++ {
		mVec = append(mVec, m[i]...)
	}
	gonumA := mat.NewDense(n, n, mVec)

	return mat.Det(gonumA)
}

func lapackInverse(m [][]float64, n int) ([][]float64, error) {
	var mVec []float64
	for i := 0; i < n; i++ {
		mVec = append(mVec, m[i]...)
	}
	gonumA := mat.NewDense(n, n, mVec)

	var inv mat.Dense
	if err := inv.Inverse(gonumA); err != nil {
		return nil, err
	}

	res :=  init2dSlice(n, n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			res[i][j] = inv.At(i, j)
		}
	}

	return res, nil
}

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

func TestLU(t *testing.T) {
	data := [][]float64{
		{7, 3, -1, 2},
		{3, 8, 1, -4},
		{-1, 1, 4, -1},
		{2, -4, -1, 6},
	}
	mat, _ := InitMat(data)

	l, u, err := LU(mat)
	if err != nil {
		t.Fatal(err)
	} else {
		a, _ := Strassen(l, u, true)
		sub, _ := MatsSub(mat, a)
		t.Logf("LU norm: %0.15f", sub.Norm(EuclideanNorm))
	}

	dataN := randData(100, 100, 1, 10)
	matN, _ := InitMat(dataN)

	lN, uN, err := LU(matN)
	if err != nil {
		t.Fatal(err)
	} else {
		a, _ := Strassen(lN, uN, true)
		sub, _ := MatsSub(matN, a)
		t.Logf("LU norm: %0.15f", sub.Norm(EuclideanNorm))
	}

	dataSym := randSymData(3, 3, 1, 10)
	matSym, _ := InitMat(dataSym)

	lSym, uSym, err := LU(matSym)
	if err != nil {
		t.Fatal(err)
	} else {
		t.Log("\n" + matSym.ToStr())
		t.Log("\n" + lSym.ToStr())
		t.Log("\n" +uSym.ToStr())
	}
}

func TestMatrix_DetLU(t *testing.T) {
	dataN := randData(10, 10, 1, 10)
	matN, _ := InitMat(dataN)

	d, err := matN.DetLU()
	if err != nil {
		t.Fatal(err)
	}

	lapack := lapackDet(dataN, len(dataN))
	dReg, err := matN.Det()
	if err != nil {
		t.Fatal(err)
	}

	t.Logf("LU Det: %0.15f; MyDet: %0.15f; LAPACK_DET: %0.15f", d, dReg, lapack)
}

func TestMatrix_InverseLU(t *testing.T) {
	dataN := randData(10, 10, 1, 10)
	matN, _ := InitMat(dataN)

	res, err := matN.InverseLU()
	if err != nil {
		t.Fatal(err)
	}

	myInv, err := matN.Inverse()
	if err != nil {
		t.Fatal(err)
	}

	lapack, err := lapackInverse(dataN, len(dataN))
	if err != nil {
		t.Fatal(err)
	}
	lapackMat, _ := InitMat(lapack)

	subLAPACK, err := MatsSub(res, lapackMat)
	if err != nil {
		t.Fatal(err)
	}
	subMy, err := MatsSub(res, myInv)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("LU inv vs. LAPACK norm: %0.15f; LU inv vs. my inv: %0.15f",
		subLAPACK.Norm(EuclideanNorm), subMy.Norm(EuclideanNorm),
	)
}