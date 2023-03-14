class Field {
  constructor(objName, blockId, conditions) {
    this.obj = blockId ? $(`#${objName}${blockId}`) : $(`#${objName}`)
    this.objName = objName
    this.blockId = blockId
    this.conditions = conditions
  }
  get value() {
    return this.obj.val()
  }
  set value(val) {
    this.obj.val(val).trigger('change')
  }
}

class NumericalField extends Field {
  get value() {
    return Number(this.obj.val())
  }
  set value(val) {
    super.value = val
  }
}

class CheckField extends Field {
  get value() {
    return this.obj.is(":checked")
  }
  set value(val) {
    this.obj.prop('checked', val).trigger('change')
  }
}

class MultiSelectField extends Field {
  get value() {
    return super.value
  }
  set value(values) {
    if (!values) {
      return
    }
    let that = this
    values.forEach((v) => {
      that.obj.children('option').each(function (i) {
        if (this.value === v) {
          this.selected = true
        }
      })
    })
    that.obj.bsMultiSelect("UpdateData")
    that.obj.trigger('change')
  }
}

class Cameras {
  get value() {
    let a = {}
    $(".cam-checkbox").each(function (i, obj) {
      if (obj.checked && !obj.disabled) {
        // let isUsePredictions = $(`#use-predictions-${obj.value}`).prop('checked')
        // the following is_use_predictions isn't used anywhere
        a[obj.value] = {is_use_predictions: true}
      }
    });
    return a
  }

  set value(cameras) {
    $(".cam-checkbox").each(function (i, obj) {
      this.checked = cameras.hasOwnProperty(obj.value)
    });
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class FieldObject {
  constructor(objName, objClass, conditions = {}) {
    this.objName = objName
    this.objClass = objClass
    this.conditions = conditions
  }

  getField(blockId = null) {
    return new this.objClass(this.objName, blockId, this.conditions)
  }
}

const AnimalIDFields = {
  get values() {
    return Object.keys(_animalIDFiels).reduce((obj, x) => {
      obj[x] = _animalIDFiels[x].getField().value
      return obj
    }, {})
  },
  set values(fieldsValues) {
    for (const [name, value] of Object.entries(fieldsValues)) {
      let obj = _animalIDFiels[name]
      if (!obj) {
        continue
      }
      obj.getField().value = value
    }
  }
}

const AllFields = {
  get values() {
    let params = Object.keys(mainFields).reduce((obj, x) => {
      obj[x] = mainFields[x].getField().value
      return obj
    }, {})
    params['blocks'] = Blocks.values
    return params
  },
  set values(fieldsValues) {
    for (const [name, value] of Object.entries(fieldsValues)) {
      if (name === 'blocks') {
        Blocks.values = value
      } else {
        let obj = mainFields[name]
        if (!obj) {
          continue
        }
        obj.getField().value = value
      }
    }
  }
}

const Blocks = {
  get values() {
    let blocks = []
    const numBlocks = mainFields.num_blocks.getField().value
    let isIdenticalBlocks = mainFields.is_identical_blocks.getField().value
    for (let i = 1; i <= numBlocks; i++) {
      let j = isIdenticalBlocks ? 1 : i
      blocks.push(new Block(j).values)
    }
    return blocks
  },
  set values(blocksValues) {
    mainFields.num_blocks.getField().value = blocksValues.length
    for (let i = 1; i <= blocksValues.length; i++) {
      let block = new Block(i)
      block.values = blocksValues[i - 1]
    }
  }
}

class Block {
  constructor(idx) {
    this.idx = idx
    this.blockType = blockFields.main.block_type.getField(this.idx).value
    this.relevantObjFields = {}
    Object.assign(this.relevantObjFields, blockFields.main)
    Object.assign(this.relevantObjFields, blockFields[this.blockType])
  }

  isConditionOk(field) {
    for (const [objName, condition] of Object.entries(field.conditions)) {
        if (!this.relevantObjFields[objName]) {
          continue
        }
        let value = this.relevantObjFields[objName].getField(this.idx).value
        if (Array.isArray(condition)) {
          if (!condition.includes(value)) {
            return false
          }
        } else if (condition !== value) {
          return false
        }
      }
    return true
  }

  getFields(isCheckCondition=true) {
    const fields = Object.keys(this.relevantObjFields).reduce((obj, x) => {
      let field = this.relevantObjFields[x].getField(this.idx)
      if (this.isConditionOk(field) || !isCheckCondition) {
        obj[x] = field
      }
      return obj
    }, {})
    return fields
  }

  get values() {
    let block = {}
    let fields = this.getFields()
    for (const [name, field] of Object.entries(fields)) {
      block[name] = field.value
      // if (name === 'reward_bugs' && !field.value) {
      //   block[name] = fields.bug_types.value
      // }
    }
    return block
  }

  set values(blockValues) {
    const fields = this.getFields(false)
    for (const [name, value] of Object.entries(blockValues)) {
      let field = fields[name]
      if (!!field) {
        field.value = value
      }
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

const mainFields = {
  // name: new FieldObject('experimentName', Field),
  animal_id: new FieldObject('animalId', Field),
  bug_types: new FieldObject('bugTypeSelect', MultiSelectField),
  time_between_blocks: new FieldObject('timeBetweenBlocks', NumericalField),
  extra_time_recording: new FieldObject('extraTimeRecording', NumericalField),
  cameras: new FieldObject('cameras', Cameras),
  num_blocks: new FieldObject('numBlocks', NumericalField),
  is_identical_blocks: new FieldObject('isIdenticalBlocks', CheckField),
  reward_bugs: new FieldObject('rewardBugSelect', MultiSelectField, {}),
  background_color: new FieldObject('backgroundColor', Field),
}

const blockFields = {
  main: {
    num_trials: new FieldObject('experimentNumTrials', NumericalField),
    trial_duration: new FieldObject('experimentTrialDuration', NumericalField),
    iti: new FieldObject('experimentITI', NumericalField),
    block_type: new FieldObject('blockTypeSelect', Field)
  },
  bugs: {
    reward_type: new FieldObject('rewardTypeSelect', Field),
    bug_speed: new FieldObject('bugSpeed', NumericalField),
    movement_type: new FieldObject('movementTypeSelect', Field),
    is_default_bug_size: new FieldObject('isDefaultBugSize', CheckField),
    bug_size: new FieldObject('bugSize', NumericalField, {is_default_bug_size: false}),
  },
  media: {
    media_url: new FieldObject('media-url', Field)
  }
}

const _animalIDFiels = {
  animal_id: new FieldObject('animalId', Field),
  sex: new FieldObject('animalSex', Field),
  exit_hole: new FieldObject('exitHolePositionSelect', Field),
  reward_any_touch_prob: new FieldObject('rewardAnyTouchProb', NumericalField),
  bug_types: new FieldObject('bugTypeSelect', MultiSelectField),
  reward_bugs: new FieldObject('rewardBugSelect', MultiSelectField, {}),
  background_color: new FieldObject('backgroundColor', Field),
}
