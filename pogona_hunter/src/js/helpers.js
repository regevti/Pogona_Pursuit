export function randomRange (min, max) {
	if (min === 0 && max === 0) {
		return 0
	}
	return Math.floor(Math.random() * (max - min) + min)
}

export const distance = (x1, y1, x2, y2) => Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2))

export const plusOrMinus = () => Math.random() < 0.5 ? -1 : 1