const timeBetweenTrial = 2000  // (ms) time takes before new ant appears after killed
const timeInEdge = 2000 // (ms) time ant spends beyond game edges
const imagesDir = 'images'

const canvas = document.querySelector('canvas')
canvas.width = 1000
canvas.height = 600
const ctx = canvas.getContext('2d')


// Load Ant Images
const getAnts = (antType, numImagesPerAnt=3) => {
	let antImages = [];
	for (let i=0;i<numImagesPerAnt;i++) {
		let img = new Image();
		img.src = `${imagesDir}/${antType}${i}.png`;
		antImages.push(img);
	}
	return antImages
}

class Ant {
	constructor(x, y, dx, dy, radius, antType, mass = 1) {
		this.mass = mass
		this.radius = radius
		this.antImages = getAnts(antType)
		this.deadImage = this.getDeadImage(antType)
		this.stepsPerImage = 10
		this.isOutEdged = false
		this.isDead = false
		this.initAnt(x, y, dx, dy)
	}

	initAnt = (x, y, dx, dy) => {
		this.x = x
		this.y = y
		this.dx = dx
		this.dy = dy
		this.currentImage = this.antImages[0]
		this.step = 0
	}

	draw = () => {
		ctx.beginPath()
		this.currentImage = this.antImages[Math.floor(this.step/this.stepsPerImage)]

		// ctx.arc(this.x, this.y, this.radius/2, Math.PI * 2, false)
		// ctx.fillStyle = this.color

		this.drawImage()
		this.step++;
		if (this.step>2*this.stepsPerImage) {this.step=0}

		ctx.fill()
		ctx.closePath()
	}

	move = particles => {
		if (this.isDead) {
			this.draw()
			return
		}
		this.edgeDetection()
		this.x += this.dx
		this.y += this.dy
		this.draw()

		for (let i = 0; i < particles.length; i++) {
			if (this === particles[i]) continue

			if (distance(this.x, this.y, particles[i].x, particles[i].y) <= this.radius + particles[i].radius) {
				this.collisionEffect(particles[i])
			}
		}
	}

	edgeTimeout = (dx, dy) => {
		this.isOutEdged = true
		let x = this.xEdge
		let y = this.yEdge
		const initTimeoutH = setTimeout(() => {
				this.initAnt(x, y, dx, dy)
				const outEdgeTimeout = setTimeout(() => {
					this.isOutEdged = false
					clearTimeout(outEdgeTimeout)
				}, 2000)
				clearTimeout(initTimeoutH)
			}, timeInEdge)
	}

	get xEdge() {
		if (this.x > canvas.width) { return canvas.width }
		else if (this.x < 0) { return 0 }
		return this.x
	}

	get yEdge() {
		if (this.y > canvas.height) { return canvas.height }
		else if (this.y < 0) { return 0 }
		return this.y
	}

	edgeDetection = () => {
		if (this.isOutEdged) {
			return
		}
		if (this.x >= canvas.width || this.x <= 0) {
			this.edgeTimeout(-this.dx, this.dy)
		}
		else if (this.y >= canvas.height || this.y <= 0) {
			this.edgeTimeout(this.dx, -this.dy)
		}
	}

	drawImage = () => {
		let antImage = this.isDead ? this.deadImage : this.currentImage
		ctx.setTransform(1,0,0,1,this.x,this.y)
    ctx.rotate(this.getAngleRadians());
		ctx.drawImage(antImage, -this.radius/2, -this.radius/2, this.radius, this.radius)
    ctx.setTransform(1,0,0,1,0,0)
	}

	getDeadImage = (antType) => {
		let img = new Image();
		img.src = `${imagesDir}/${antType}_dead.png`;
		return img
	}

	rotate = (dx, dy, angle) => {
		return {
			dx: dx * Math.cos(angle) - dy * Math.sin(angle),
			dy: dx * Math.sin(angle) + dy * Math.cos(angle)
		}
	}

	getAngleRadians = () => {
		return Math.atan2(this.dy, this.dx)  + Math.PI/2
	}

	collisionEffect = (otherAnt) => {
		// collision between 2 ants
		const angle = -Math.atan2(otherAnt.y - this.y, otherAnt.x - this.x)
		const u1 = this.rotate(this.dx, this.dy, angle)
		const u2 = this.rotate(otherAnt.dx, otherAnt.dy, angle)

		const v1 = {
			dx: ((this.mass - otherAnt.mass) * u1.dx / (this.mass + otherAnt.mass)) + (2 * otherAnt.mass * u2.dx / (this.mass + otherAnt.mass)),
			dy: u1.dy
		}

		const v2 = {
			dx: ((this.mass - otherAnt.mass) * u2.dx / (this.mass + otherAnt.mass)) + (2 * otherAnt.mass * u1.dx / (this.mass + otherAnt.mass)),
			dy: u2.dy
		}

		const rotatedv1 = this.rotate(v1.dx, v1.dy, -angle)
		const rotatedv2 = this.rotate(v2.dx, v2.dy, -angle)

		this.dx = rotatedv1.dx
		this.dy = rotatedv1.dy

		otherAnt.dx = rotatedv2.dx
		otherAnt.dy = rotatedv2.dy
	}
}


const randomRange = (min, max) => {
	while (true) {
		let randomNum = Math.floor(Math.random() * (max - min) + min)
		if (randomNum !== 0)
			return randomNum
	}
}

const randomColor = () => {
	let r = randomRange(0, 255)
	let g = randomRange(0, 255)
	let b = randomRange(0, 255)
	let a = Math.random() * (1 - 0.3) + 0.3 // -> for value between 0 - 1

	return `rgba(${r}, ${g}, ${b}, ${a}`
}

const distance = (x1, y1, x2, y2) => Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2))

let ants = []
let animationHandler = null

const spawnAnts = noOfAnts => {
	for (let i = 0; i < noOfAnts; i++) {
		let radius = randomRange(70, 90)
		let x = randomRange(radius, canvas.width - radius)
		let y = randomRange(radius, canvas.height - radius)
		const properties = [ // -> x, y, dx, dy, radius, antType, mass
			x,
			y,
			randomRange(-2, 2),
			randomRange(-2, 2),
			radius,
			'cockroach',
		]

		if (i !== 0) {
			for (let j = 0; j < i; j++) {
				let d = distance(x, y, ants[j].x, ants[j].y)
				if (d <= radius + ants[j].radius) {
					x = randomRange(radius, canvas.width - radius)
					y = randomRange(radius, canvas.height - radius)
					j = -1
				}
			}
		}

		ants.push(new Ant(...properties))
	}
}

let timeoutH = 0

const destruct = antIndex => {
	ants[antIndex].isDead = true
	updateScore(++score)
	const killTimeout = setTimeout(() => {
		ants = ants.filter((items, index) => antIndex !== index)
		if (ants.length === 0)
			timeoutH = setTimeout(() => {
				cancelAnimationFrame(animationHandler)
				init()
				clearTimeout(timeoutH)
			}, timeBetweenTrial)
	}, 2000)
}


canvas.addEventListener('mousedown', event => {
	let x = event.x;
	let y = event.y;

	x -= canvas.offsetLeft;
	y -= canvas.offsetTop;

	console.log(x, y)

	for (let i = 0; i < ants.length; i++) {
		if (distance(x, y, ants[i].x, ants[i].y) <= ants[i].radius/2) {
			destruct(i)
		}
	}
})

function animate() {
	animationHandler = requestAnimationFrame(animate)
	ctx.clearRect(0, 0, canvas.width, canvas.height)
	ants.forEach(ant => ant.move(ants))
}

function init() {
	ants = []
	spawnAnts(1)
	animate()
}

init()