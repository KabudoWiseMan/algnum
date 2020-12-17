package algnum

func init2dSlice(rows, cols int) [][]float64 {
	data := make([][]float64, rows)
	for i := range data {
		data[i] = make([]float64, cols)
	}
	return data
}

func copy2dSlice(src [][]float64) [][]float64 {
	dst := make([][]float64, len(src))
	for i := range dst {
		dst[i] = make([]float64, len(src[i]))
		copy(dst[i], src[i])
	}

	return dst
}

func isPowerOfTwo(x int) bool {
	return (x != 0) && ((x & (x - 1)) == 0)
}

func intMax(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func intMax4(a, b, c, d int) int {
	return intMax(intMax(intMax(a, b), c), d)
}

func nearestPowerOfTwo(a int) int {
	if isPowerOfTwo(a) {
		return a
	}
	count := 0
	for a != 0 {
		a = a >> 1
		count++
	}
	return 1 << count
}

func findMinMax(a []float64) (float64, float64) {
	min, max := a[0], a[0]

	for _, ai := range a {
		if ai < min {
			min = ai
		}
		if ai > max {
			max = ai
		}
	}

	return min, max
}