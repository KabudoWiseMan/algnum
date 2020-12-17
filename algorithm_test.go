package algnum

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
	"math"
	"math/rand"
	"testing"
	"time"
)

func lapackSolve(m [][]float64, n int, f []float64) ([]float64, error) {
	var mVec []float64
	for i := 0; i < n; i++ {
		mVec = append(mVec, m[i]...)
	}
	gonumA := mat.NewDense(n, n, mVec)
	gonumF := mat.NewVecDense(n, f)
	result := mat.NewVecDense(n, nil)

	err := result.SolveVec(gonumA, gonumF)
	if err != nil {
		return nil, err
	}
	return result.RawVector().Data, nil
}

func randData(rows, cols int, min, max int) [][]float64 {
	rand.Seed(time.Now().UnixNano())

	res := make([][]float64, rows)
	for i := range res {
		res[i] = make([]float64, cols)
	}

	for i := range res {
		for j := range res {
			res[i][j] = float64(rand.Intn(max - min) + min)
		}
	}

	return res
}

func randSymData(rows, cols int, min, max int) [][]float64 {
	rand.Seed(time.Now().UnixNano())

	res := make([][]float64, rows)
	for i := range res {
		res[i] = make([]float64, cols)
	}

	for i := range res {
		for j := 0; j <= i; j++ {
			r := float64(rand.Intn(max - min) + min)
			res[i][j] = r
			res[j][i] = r
		}
	}

	return res
}

func randFree(dim int, min, max int) []float64 {
	rand.Seed(time.Now().UnixNano())

	res := make([]float64, dim)
	for i := range res {
		res[i] = float64(rand.Intn(max - min) + min)
	}

	return res
}

// each diagonal element will be >= sum of other elements in the row
func randDiagDominantData(rows, cols int, min, max int) [][]float64 {
	rand.Seed(time.Now().UnixNano())

	res := make([][]float64, rows)
	for i := range res {
		res[i] = make([]float64, cols)
	}

	for i := range res {
		for j := range res {
			res[i][j] = float64(rand.Intn(max - min) + min)
		}
	}

	for i := 0; i < rows; i++ {
		var sum float64
		for j := 0; j < cols; j++ {
			if j == i {
				continue
			}
			sum += math.Abs(res[i][j])
		}
		if math.Abs(res[i][i]) < sum {
			res[i][i] = sum * 10
		}
	}

	return res
}

func randDiagSym(rows, cols int, min, max int) [][]float64 {
	rand.Seed(time.Now().UnixNano())

	res := make([][]float64, rows)
	for i := range res {
		res[i] = make([]float64, cols)
	}

	for i := range res {
		for j := 0; j <= i; j++ {
			r := float64(rand.Intn(max - min) + min)
			res[i][j] = r
			res[j][i] = r
		}
	}

	for i := 0; i < rows; i++ {
		var sum float64
		for j := 0; j < cols; j++ {
			if j == i {
				continue
			}
			sum += math.Abs(res[i][j])
		}
		if math.Abs(res[i][i]) < sum {
			res[i][i] = sum * 10
		}
	}

	return res
}

func TestGauss(t *testing.T) {
	data := [][]float64{
		{3, 2, 1},
		{1, 3, 2},
		{1, 2, 4},
	}
	mat, _ := InitMat(data)
	f := []float64{1, 4, 2}

	expectedRes := []float64{-float64(14)/float64(19), float64(32)/float64(19), -float64(3)/float64(19)}

	res, err := Gauss(mat, f)
	if err != nil {
		t.Fatal(err)
	} else if !VectsEq(expectedRes, res, Epsilon) {
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
	} else if !VectsEq(expectedRes2, res2, Epsilon) {
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
	} else if !VectsEq(expectedRes3, res3, Epsilon) {
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
		t.Fatal("result is wrong, input:\nA = ", mat4.ToStr(), "\nf = ", VectToStr(f4), "\nexpected error, got:\n", VectToStr(res4))
	}

	dataN := randData(100, 100, 1, 1000)
	matN, _ := InitMat(dataN)
	fN := randFree(100, 1, 1000)

	resN, err := Gauss(matN, fN)
	if err != nil {
		t.Fatal(err)
	}
	check, err := MatVecMul(matN, resN)
	if err != nil {
		t.Fatal(err)
	} else if !VectsEq(fN, check, Epsilon) {
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
	} else if !VectsEq(expectedRes5, res5, Epsilon) {
		t.Fatalf("result is wrong: expected\n %s,\ngot\n %s", VectToStr(expectedRes5), VectToStr(res5))
	} else {
		t.Log("gauss works correct, input:\nA = ", mat5.ToStr(), "\nf = ", VectToStr(f5), "\nresult:", VectToStr(res5))
	}
}

func TestPerturbations2(t *testing.T) {
	data := [][]float64{
		{100, 99},
		{99, 98},
	}
	A, _ := InitMat(data)

	dataDA := [][]float64{
		{0, 0},
		{0, 0},
	}
	dA, _ := InitMat(dataDA)

	f := []float64{199, 197}
	df := []float64{-0.01, 0.01}

	x, err := Gauss(A, f)
	if err != nil {
		t.Fatal(err)
	}
	t.Log("x:", VectToStr(x))

	APlusDA, _ := MatsSum(A, dA)
	fPlusDf, _ := VecsSum(f, df)
	xPlusDx, err := Gauss(APlusDA, fPlusDf)
	if err != nil {
		t.Fatal(err)
	}
	t.Log("x + dx:", VectToStr(xPlusDx))

	dx, _ := VecsSub(xPlusDx, x)
	t.Log("dx:", VectToStr(dx))

	invA, err := A.Inverse()
	if err != nil {
		t.Fatal(err)
	}
	t.Log("inverse A:\n", invA.ToStr())

	normA := A.Norm(InfinityNorm)

	t.Logf("||dx|| / ||x|| = %0.15f", VecNorm(dx, InfinityNorm) / VecNorm(x, InfinityNorm))
	muA := invA.Norm(InfinityNorm) * normA
	t.Logf("μA = %0.15f", muA)
	t.Logf("μA * (||df|| / ||f|| + ||dA|| / ||A||) = %0.15f", muA * (VecNorm(df, InfinityNorm) / VecNorm(f, InfinityNorm) + dA.Norm(InfinityNorm) / normA))
}

func TestPerturbations(t *testing.T) {
	data := randData(10, 10, 1, 100)
	A, _ := InitMat(data)
	t.Log("A:\n", A.ToStr())

	dataDA := randData(10, 10, 1, 5)
	dA, _ := InitMat(dataDA)
	t.Log("dA:\n", dA.ToStr())

	f := randFree(10, 1, 100)
	t.Log("f:", VectToStr(f))
	df := randFree(10, 1, 5)
	t.Log("df:", VectToStr(df))

	x, err := Gauss(A, f)
	if err != nil {
		t.Fatal(err)
	}
	t.Log("x:", VectToStr(x))

	APlusDA, _ := MatsSum(A, dA)
	fPlusDf, _ := VecsSum(f, df)
	xPlusDx, err := Gauss(APlusDA, fPlusDf)
	if err != nil {
		t.Fatal(err)
	}
	t.Log("x + dx:", VectToStr(xPlusDx))

	dx, _ := VecsSub(xPlusDx, x)
	t.Log("dx:", VectToStr(dx))

	invA, err := A.Inverse()
	if err != nil {
		t.Fatal(err)
	}
	t.Log("inverse A:\n", invA.ToStr())

	normA := A.Norm(EuclideanNorm)

	t.Logf("||dx||2 / ||x||2 = %0.15f", VecNorm(dx, EuclideanNorm) / VecNorm(x, EuclideanNorm))
	muA := invA.Norm(EuclideanNorm) * normA
	t.Logf("μA2 = %0.15f", muA)
	t.Logf("μA2 * (||df||2 / ||f||2 + ||dA||2 / ||A||2) = %0.15f", muA * (VecNorm(df, EuclideanNorm) / VecNorm(f, EuclideanNorm) + dA.Norm(EuclideanNorm) / normA))

	normAinf := A.Norm(InfinityNorm)

	t.Logf("||dx||∞ / ||x||∞ = %0.15f", VecNorm(dx, InfinityNorm) / VecNorm(x, InfinityNorm))
	muAinf := invA.Norm(InfinityNorm) * normAinf
	t.Logf("μA∞ = %0.15f", muAinf)
	t.Logf("μA∞ * (||df||∞ / ||f||∞ + ||dA||∞ / ||A||∞) = %0.15f", muAinf * (VecNorm(df, InfinityNorm) / VecNorm(f, InfinityNorm) + dA.Norm(InfinityNorm) / normAinf))
}

func TestJacobi(t *testing.T) {
	data := [][]float64{
		{3, 2, 1},
		{1, 3, 2},
		{1, 2, 4},
	}
	mat, _ := InitMat(data)
	f := []float64{1, 4, 2}

	expectedRes := []float64{-float64(14) / float64(19), float64(32) / float64(19), -float64(3) / float64(19)}

	res, err := Jacobi(mat, f)
	if err != nil {
		t.Fatal(err)
	} else if !VectsEq(expectedRes, res, Epsilon) {
		t.Fatalf("result is wrong: expected\n %s,\ngot\n %s", VectToStr(expectedRes), VectToStr(res))
	} else {
		t.Log("jacobi works correct, input:\nA = ", mat.ToStr(), "\nf = ", VectToStr(f), "\nresult:", VectToStr(res))
	}

	data2 := [][]float64{
		{100, 30, -70},
		{15, -50, -5},
		{6, 2, 20},
	}
	mat2, _ := InitMat(data2)
	f2 := []float64{60, -40, 28}

	expectedRes2 := []float64{1, 1, 1}

	res2, err := Jacobi(mat2, f2)
	if err != nil {
		t.Fatal(err)
	} else if !VectsEq(expectedRes2, res2, Epsilon) {
		t.Fatalf("result is wrong: expected\n %s,\ngot\n %s", VectToStr(expectedRes2), VectToStr(res2))
	} else {
		t.Log("jacobi works correct, input:\nA = ", mat2.ToStr(), "\nf = ", VectToStr(f2), "\nresult:", VectToStr(res2))
	}

	dataN := randDiagDominantData(1000, 1000, 1, 100)
	matN, _ := InitMat(dataN)
	fN := randFree(1000, 1, 1000)

	resN, err := Jacobi(matN, fN)
	if err != nil {
		t.Fatal(err)
	}
	check, err := MatVecMul(matN, resN)
	if err != nil {
		t.Fatal(err)
	} else if !VectsEq(fN, check, 1e-1) {
		t.Fatalf("result is wrong, input:\nA = %s\nf = %s\nres = %s\ncheck:%s", matN.ToStr(), VectToStr(fN), VectToStr(resN), VectToStr(check))
	} else {
		t.Log("jacobi works correct, input:\nA = ", matN.ToStr(), "\nf = ", VectToStr(fN), "\nresult:", VectToStr(resN), "\ncheck:", VectToStr(check))
	}
}

func TestSeidel(t *testing.T) {
	data := [][]float64{
		{3, 2, 1},
		{1, 3, 2},
		{1, 2, 4},
	}
	mat, _ := InitMat(data)
	f := []float64{1, 4, 2}

	expectedRes := []float64{-float64(14) / float64(19), float64(32) / float64(19), -float64(3) / float64(19)}

	res, err := Seidel(mat, f)
	if err != nil {
		t.Fatal(err)
	} else if !VectsEq(expectedRes, res, Epsilon) {
		t.Fatalf("result is wrong: expected\n %s,\ngot\n %s", VectToStr(expectedRes), VectToStr(res))
	} else {
		t.Log("siedel works correct, input:\nA = ", mat.ToStr(), "\nf = ", VectToStr(f), "\nresult:", VectToStr(res))
	}

	data2 := [][]float64{
		{100, 30, -70},
		{15, -50, -5},
		{6, 2, 20},
	}
	mat2, _ := InitMat(data2)
	f2 := []float64{60, -40, 28}

	expectedRes2 := []float64{1, 1, 1}

	res2, err := Seidel(mat2, f2)
	if err != nil {
		t.Fatal(err)
	} else if !VectsEq(expectedRes2, res2, Epsilon) {
		t.Fatalf("result is wrong: expected\n %s,\ngot\n %s", VectToStr(expectedRes2), VectToStr(res2))
	} else {
		t.Log("seidel works correct, input:\nA = ", mat2.ToStr(), "\nf = ", VectToStr(f2), "\nresult:", VectToStr(res2))
	}

	dataN := randDiagDominantData(1000, 1000, 1, 100)
	matN, _ := InitMat(dataN)
	fN := randFree(1000, 1, 1000)

	resN, err := Seidel(matN, fN)
	if err != nil {
		t.Fatal(err)
	}
	check, err := MatVecMul(matN, resN)
	if err != nil {
		t.Fatal(err)
	} else if !VectsEq(fN, check, 1e-2) {
		t.Fatalf("result is wrong, input:\nA = %s\nf = %s\nres = %s\ncheck:%s", matN.ToStr(), VectToStr(fN), VectToStr(resN), VectToStr(check))
	} else {
		t.Log("seidel works correct, input:\nA = ", matN.ToStr(), "\nf = ", VectToStr(fN), "\nresult:", VectToStr(resN), "\ncheck:", VectToStr(check))
	}
}

func TestStrassen(t *testing.T) {
	aData := randData(3000, 3000, 1, 10)
	a, _ := InitMat(aData)

	bData := randData(3000, 3000, 1, 10)
	b, _ := InitMat(bData)

	startStrass := time.Now()
	res, err := Strassen(a, b, true)
	stopStrass := time.Since(startStrass)
	start := time.Now()
	expectedRes, _ := MatMul(a, b)
	stop := time.Since(start)

	if err != nil {
		t.Fatal(err)
	} else if !MatsEq(res, expectedRes, Epsilon) {
		t.Fatalf("result is wrong: expected\n %s,\ngot\n %s", expectedRes.ToStr(), res.ToStr())
	} else {
		t.Logf("strassen works correct, time: %s vs. %s\n", stop, stopStrass)
		//t.Log(res.ToStr())
	}
}

func TestCholesky(t *testing.T) {
	data := randDiagSym(100, 100, 1, 100)
	mat, _ := InitMat(data)
	f := randFree(100, 1, 100)

	l, err := Cholesky(mat)
	if err != nil {
		t.Fatal(err)
	}

	y, err := GaussClassic(l, f)
	if err != nil {
		t.Fatal(err)
	}

	lt := TransposeMat(l)

	res, err := GaussClassic(lt, y)
	if err != nil {
		t.Fatal(err)
	}
	check, err := MatVecMul(mat, res)
	if err != nil {
		t.Fatal(err)
	} else if !VectsEq(f, check, 1e-2) {
		t.Fatalf("result is wrong, input:\nA = %s\nf = %s\nres = %s\ncheck:%s", mat.ToStr(), VectToStr(f), VectToStr(res), VectToStr(check))
	} else {
		t.Log("seidel works correct, input:\nA = ", mat.ToStr(), "\nf = ", VectToStr(f), "\nresult:", VectToStr(res), "\ncheck:", VectToStr(check))
	}
}

func TestMinMaxEigenvalues(t *testing.T) {
	aData := [][]float64 {
		{7, 0.5},
		{0.5, 1},
	}
	a, _ := InitMat(aData)

	min, max, err := MinMaxEigenvalues(a)
	if err != nil {
		t.Fatal(err)
	} else {
		t.Log(min, max)
	}

	rData := randDiagSym(100, 100, 1, 10)
	r, _ := InitMat(rData)

	minR, maxR, err := MinMaxEigenvalues(r)
	if err != nil {
		t.Fatal(err)
	} else {
		t.Log(minR, maxR)
	}
}

func TestFixedPointIteration(t *testing.T) {
	data := [][]float64{
		{3, 1, 1},
		{1, 3, 2},
		{1, 2, 4},
	}
	mat, _ := InitMat(data)
	f := []float64{1, 4, 2}

	expectedRes := []float64{-0.0952, 1.52, -0.238}

	res, err := FixedPointIteration(mat, f)
	if err != nil {
		t.Fatal(err)
	} else if !VectsEq(expectedRes, res, 1e-2) {
		t.Fatalf("result is wrong: expected\n %s,\ngot\n %s", VectToStr(expectedRes), VectToStr(res))
	} else {
		t.Log("fixed-point iteration works correct, input:\nA = ", mat.ToStr(), "\nf = ", VectToStr(f), "\nresult:", VectToStr(res))
	}

	dataN := randDiagSym(1000, 1000, 1, 10)
	matN, _ := InitMat(dataN)
	fN := randFree(1000, 1, 10)

	resN, err := FixedPointIteration(matN, fN)
	if err != nil {
		t.Fatal(err)
	}
	check, err := MatVecMul(matN, resN)
	if err != nil {
		t.Fatal(err)
	} else if !VectsEq(fN, check, 1e-2) {
		t.Fatalf("result is wrong, input:\nA = %s\nf = %s\nres = %s\ncheck:%s", matN.ToStr(), VectToStr(fN), VectToStr(resN), VectToStr(check))
	} else {
		t.Log("fixed-point iteration works correct, input:\nA = ", matN.ToStr(), "\nf = ", VectToStr(fN), "\nresult:", VectToStr(resN), "\ncheck:", VectToStr(check))
	}
}

func TestMyMinMaxEigenvalues(t *testing.T) {
	aData := [][]float64 {
		{7, 0.5},
		{0.5, 1},
	}
	a, _ := InitMat(aData)

	min, max, err := MyMinMaxEigenvalues(a)
	if err != nil {
		t.Fatal(err)
	} else {
		t.Log(min, max)
	}

	rData := randDiagSym(1000, 1000, 1, 10)
	r, _ := InitMat(rData)

	start := time.Now()
	minR, maxR, err := MinMaxEigenvalues(r)
	if err != nil {
		t.Fatal(err)
	} else {
		t.Log(minR, maxR)
	}
	stop := time.Since(start)

	startMy := time.Now()
	minR, maxR, err = MyMinMaxEigenvalues(r)
	if err != nil {
		t.Fatal(err)
	} else {
		t.Log(minR, maxR)
	}
	stopMy := time.Since(startMy)

	fmt.Println(stop.Seconds(), stopMy.Seconds())
}

func TestLUSolve(t *testing.T) {
	dataN := randData(100, 100, 1, 10)
	matN, _ := InitMat(dataN)
	fN := randFree(100, 1, 10)

	l, u, err := LU(matN)
	if err != nil {
		t.Fatal(err)
	}

	resN, err := LUSolve(l, u, fN)
	if err != nil {
		t.Fatal(err)
	}

	lapack, err := lapackSolve(dataN, len(dataN), fN)
	if err != nil {
		t.Fatal(err)
	}

	sub, _ := VecsSub(resN, lapack)
	n := VecNorm(sub, EuclideanNorm)
	t.Logf("LU-solve norm: %0.15f", n)
}