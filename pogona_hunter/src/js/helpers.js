export function randomRange (min, max) {
	if (min === 0 && max === 0) {
		return 0
	}
	return Math.floor(Math.random() * (max - min) + min)
}

export const distance = (x1, y1, x2, y2) => Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2))

export const plusOrMinus = () => Math.random() < 0.5 ? -1 : 1

export function randBM() {
    let u = 0, v = 0
    while(u === 0) u = Math.random(); //Converting [0,1) to (0,1)
    while(v === 0) v = Math.random();
    return Math.sqrt( -2.0 * Math.log( u ) ) * Math.cos( 2.0 * Math.PI * v );
}