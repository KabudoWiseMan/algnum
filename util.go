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
