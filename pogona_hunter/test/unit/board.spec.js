import {mount} from "@vue/test-utils"
import Board from './../../src/components/board'

describe('App', () => {
  // Inspect the raw component options
  it('has data', () => {
    expect(typeof Board.data).toBe('function')
  })
})
