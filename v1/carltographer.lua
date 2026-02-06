--[[ Carltographer: procedural 40k terrain generation in Tabletop Simulator

Vague feature ideas for the future:
* Some kind of UI for configuration
* If rotationally symmetrical, configurable chance of a centerpiece terrain feature (assuming it can be rotationally symmetrical itself)
* Some kind of saving and loading of layouts, and ability to refine existing layouts
* Some presets, such as from WTC and Leviathan and old US Open / GW layout
* Different themes: manufactorum theme, forest ruins theme, etc.
* Some sort of optimization for an amount of line of sight blocking to and from certain places (a much more difficult feature, and might be CPU taxing)
* spawning gameplay-irrelevant cool doodads/greebles: debris, grass/bushes, etc.

Vague sketch of desired behavior:
Given a Map, can iterate on it by making a copy with some random mutation applied.
Then, we check for validity, and go back if it is invalid (e.g. things too close or overlapping).
Then, check the goodness rating (however that gets calculated), and *possibly* go back if worse.
Uncertain if the generation mechanism should do multiple iterations speculatively before checking goodness, or have more complicated mutations. I can imagine things getting stuck in a not very good local optimum for a while. Basically genetic algorithms stuff; will take some experimentation.

Possible mutations:
* Add a terrain feature (pick one at random somehow, then put it in some location, maybe nudge it if it overlaps others?)
* Remove a terrain feature
* Replace a terrain feature? Also there might exist some mutation logic for some terrain features hypothetically (such as changing size, or changing up windows in a wall, etc).
* Move a terrain feature (maybe some options here like move closer to the closest other feature, or go in a random direction, or spread out)
* Rotate a terrain feature (this might have constraints like only picking from right angles for some places; in general, probably only want to allow whole number y axis rotation, which may have the upside of potentially allowing a lookup table for trig functions :shrug:)
* After rotating or replacing a terrain feature, we can probably automatically adjust it a bit if it is only slightly invalid: e.g., if it now encroaches on another terrain feature, but there is space to move, slide it away from the other terrain feature. This might get complicated to do in full generality unfortunately.

Possible crossover functions?
* Maybe take a shuffled mix of terrain pieces from each individual, adding each iteratively so long as the addition would be valid

Possible goodness checks (aside from hard validity checks like terrain overlapping or gaps for mobility):
* Based on peoples' preferences on number of various terrain features of such and such types
* Line of Sight testing to some points on objective markers or one's DZ? Can do this by checking for collision along a line segment maybe. There may be some preference for amount of relatively safe space in one's DZ, or on midfield objectives, etc. This is probably the hardest and most CPU taxing.

--]]

-- Some constants
mainObjectGUID = '7041e2'
matSurfaceGUIDs = {'4ee1f2','blahblahblah'} -- I don't know what other surface GUIDs there might be, so it's an array.
spawnedTag = 'Terrain Object Spawned by Carltographer'
matYOffset = 0.96

-- Saved data
centralCarltographer = nil

-- making local some things for stupid lua performance reasons

local sin = math.sin
local cos = math.cos
local abs = math.abs
local floor = math.floor
local max = math.max
local min = math.min
local rad = math.rad
local sqrt = math.sqrt
local insert = table.insert

-- end locals

-- begin "class.lua" from http://lua-users.org/wiki/SimpleLuaClasses

function class(base, init)
   local c = {}    -- a new class instance
   if not init and type(base) == 'function' then
      init = base
      base = nil
   elseif type(base) == 'table' then
    -- our new class is a shallow copy of the base class!
      for i,v in pairs(base) do
         c[i] = v
      end
      c._base = base
   end
   -- the class will be the metatable for all its objects,
   -- and they will look up their methods in it.
   c.__index = c

   -- expose a constructor which can be called by <classname>(<args>)
   local mt = {}
   mt.__call = function(class_tbl, ...)
   local obj = {}
   setmetatable(obj,c)
   if init then
      init(obj,...)
   else 
      -- make sure that any stuff from the base class is initialized!
      if base and base.init then
      base.init(obj, ...)
      end
   end
   return obj
   end
   c.init = init
   c.is_a = function(self, klass)
      local m = getmetatable(self)
      while m do 
         if m == klass then return true end
         m = m._base
      end
      return false
   end
   setmetatable(c, mt)
   return c
end

-- end "class.lua"

-- Begin utilities "library"

function shallowCopyTable(original)
  local o = {}
  for k, v in pairs(original or {}) do
    o[k] = v
  end
  return o
end

function shallowCopyArray(original)
  local o = {}
  for k, v in ipairs(original or {}) do
    o[k] = v
  end
  return o
end

function arrayContains(arr, elem)
  for _, thing in ipairs(arr) do
    if thing == elem then
      return true
    end
  end
  return false
end

-- End utilities "library"

-- Begin geometry "library"

-- returns the square of the distance between two 2D points (x1, y1) and (x2, y2)
-- Only returning the square of the distance to save on square root math time (in case that matters)
function dist2DSquared(x1, y1, x2, y2)
  return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)
end
-- returns the distance between two 2D points (x1, y1) and (x2, y2). Because
-- of the square root, may be a bit computationally expensive.
function dist2D(x1, y1, x2, y2)
  return sqrt(dist2DSquared(x1, y1, x2, y2))
end

-- returns the square of the distance between 2D line segment ((x1, y1), (x2, y2)) and point (x3, y3).
-- following the algorithm at https://paulbourke.net/geometry/pointlineplane/
function dist2DSegmentPointSquared(x1, y1, x2, y2, x3, y3)
  if x1 == x2 and y1 == y2 then
    return dist2DSquared(x1, y1, x3, y3)
  end
  local u = ((x3 - x1) * (x2 - x1) + (y3 - y1) * (y2 - y1)) / dist2DSquared(x1, y1, x2, y2)
  if u > 0.0 and u < 1.0 then
    -- (x4, y4) is the perpendicular intercept
    local x4 = x1 +  u * (x2 - x1)
    local y4 = y1 +  u * (y2 - y1)
    return dist2DSquared(x4, y4, x3, y3)
  end
  return min(dist2DSquared(x1, y1, x3, y3), dist2DSquared(x2, y2, x3, y3)) 
end
function dist2DSegmentPoint(x1, y1, x2, y2, x3, y3)
  return sqrt(dist2DSegmentPointSquared(x1, y1, x2, y2, x3, y3))
end

-- returns true if there is intersection between two line segments ((x1, y1), (x2, y2)) and ((x3, y3), (x4, y4))
-- following the algorithm at https://paulbourke.net/geometry/pointlineplane/
function collide2DSegments(x1, y1, x2, y2, x3, y3, x4, y4)
  local denominator = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
  if denominator == 0 then -- parallel
    return false
  end
  local ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denominator
  if ua <= 0 or ua >= 1 then
    return false
  end
  local ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denominator
  return ub > 0 and ub < 1
end

--[[ A representation of a transformation in 3D space.

This uses the same format as the TTS object transform, so that it can be used directly when instantiating/moving an object.

posX/posY/posZ are position in 3D space
rotX/rotY/rotZ are rotation in *degrees* (note not radians, when the math library wants radians)
scaleX/scaleY/scaleZ are stretching

This represents first scaling an object by `scale``, then rotating by `rot, then moving by `pos`.
--]]
Transform = class(function(o, args)
  args = args or {}
  o.posX = args.posX or 0
  o.posY = args.posY or 0
  o.posZ = args.posZ or 0
  o.rotX = args.rotX or 0
  o.rotY = args.rotY or 0
  o.rotZ = args.rotZ or 0
  o.scaleX = args.scaleX or 1
  o.scaleY = args.scaleY or 1
  o.scaleZ = args.scaleZ or 1
end)

-- Create a new Transform representing what would happen if you took a thing through my Transform followed by the other Transform.
--
-- WARNING: only supporting combining rotations by `other` 's Y axis, as that's all that will happen in real use here.
function Transform:combine(other)
  if getmetatable(other) ~= Transform then
    other = Transform(other)
  end
  local result = Transform(self)
  -- scale `result`
  result.scaleX = result.scaleX * other.scaleX
  result.scaleY = result.scaleY * other.scaleY
  result.scaleZ = result.scaleZ * other.scaleZ
  result.posX = result.posX * other.scaleX
  result.posY = result.posY * other.scaleY
  result.posZ = result.posZ * other.scaleZ

  -- rotate it about its current origin
  -- WARNING: only supporting repositioning via rotation about Y axis.
  -- Notes on TTS angles and positioning:
  -- Viewed from above, increasing Y rotation number rotates clockwise.
  -- For geometry of rotation purposes, treat X like the X axis, and Z like the Y axis.
  result.rotX = result.rotX + other.rotX
  result.rotY = result.rotY + other.rotY
  result.rotZ = result.rotZ + other.rotZ
  if other.rotY ~= 0 then
    local thetaRadians = rad(-other.rotY)
    local newX = result.posX * cos(thetaRadians) - result.posZ * sin(thetaRadians)
    local newZ = result.posX * sin(thetaRadians) + result.posZ * cos(thetaRadians)
    result.posX = newX
    result.posZ = newZ
  end
  
  --reposition
  result.posX = result.posX + other.posX
  result.posY = result.posY + other.posY
  result.posZ = result.posZ + other.posZ
  
  return result
end

function Transform:clone()
  return Transform(self)
end

function rectDistanceToBounds(transform, matX, matZ)
  local dist = matX + matZ
  -- for a corner, consider it as a point at e.g. (+/1 0.5, +/1 0.5), then apply our transform
  for dX = 0, 1 do
    for dZ = 0, 1 do
      local t = Transform({posX = dX - 0.5, posZ = dZ - 0.5})
      t = t:combine(transform)
      dist = min(matX * 0.5 - abs(t.posX), dist)
      dist = min(matZ * 0.5 - abs(t.posZ), dist)
    end
  end
  return dist
end

function rectDistance(t1, t2)
  -- initialize distance (squared) with the centers, just to have a starting high number.
  local distSquared = dist2DSquared(t1.posX, t1.posZ, t2.posX, t2.posZ)
  local ts = {t1, t2}

  local corners = {{}, {}}
  local cornerOrder = {{0.5, 0.5}, {0.5, -0.5}, {-0.5, -0.5}, {-0.5, 0.5}}
  for i = 1, 2 do
    for _, cornerOffset in ipairs(cornerOrder) do
      local t = Transform({posX = cornerOffset[1], posZ = cornerOffset[2]})
      t = t:combine(ts[i])
      insert(corners[i], t)
    end
  end

  local edges = {{}, {}}
  for i = 1, 2 do
    for j = 1, 4 do
      insert(edges[i], {corners[i][j], corners[i][j % 4 + 1]})
    end
  end
  
  -- check for edge overlaps
  for _, edge1 in ipairs(edges[1]) do
    for _, edge2 in ipairs(edges[2]) do
      if collide2DSegments(
          edge1[1].posX, edge1[1].posZ, edge1[2].posX, edge1[2].posZ,
          edge2[1].posX, edge2[1].posZ, edge2[2].posX, edge2[2].posZ) then
        return 0
      end
    end
  end
  
  -- get the distance between edges and corners
  
  for i = 1, 2 do
    for j = 1, 4 do
      for k = 1, 4 do
        local corner = corners[i][j]
        local edge = edges[i % 2 + 1][k]
        distSquared = min(
          distSquared,
          dist2DSegmentPointSquared(
            edge[1].posX, edge[1].posZ, edge[2].posX, edge[2].posZ,
            corner.posX, corner.posZ))
      end
    end
  end
  
  return sqrt(distSquared)
end

-- End geometry "library"

-- Begin randomness "library"

RandomGen = class()

function RandomGen:shuffle(arr)
  for i = #arr, 2, -1 do
    local j = self:intRange(1, i)
    arr[i], arr[j] = arr[j], arr[i]
  end
  return arr
end

function RandomGen:pick(arr)
  if #arr < 1 then return nil end
  return arr[self:intRange(1, #arr)]
end

-- given [(thing, weight)], pick a thing, weighted by weights
function RandomGen:weightedPick(arr)
  local totalWeight = 0.0
  for _, p in ipairs(arr) do
    totalWeight = totalWeight + p[2]
  end
  if totalWeight <= 0.0 then
    return nil
  end
  local randPick = self:floatRange(0.0, totalWeight)
  for _, p in ipairs(arr) do
    randPick = randPick - p[2]
    if randPick < 0.0 then
      return p[1]
    end
  end
  return arr[#arr][1]
end

RealRandomGen = class(RandomGen)

-- returns a random integer in [startInt, endInt] (note that it is inclusive)
function RealRandomGen:intRange(startInt, endInt)
  return math.random(startInt, endInt)
end

-- returns a random floating point number in [startFloat, endFloat)
function RealRandomGen:floatRange(startFloat, endFloat)
  return math.random() * (endFloat - startFloat) + startFloat
end

-- End randomness "library"

-- Begin genetic algorithm section

GAIndividual = class(function(o, args)
  args = args or {}
  -- The thing that we are optimizing.
  o.thing = args.thing or nil
  -- From oldest to newest scores. On initial creation, this will be empty.
  o.scoreHistory = shallowCopyArray(args.scoreHistory)
end)

function GAIndividual:clone()
  return GAIndividual(self)
end
function GAIndividual:latestScore()
  return self.scoreHistory[#self.scoreHistory]
end

GAOptions = class(function(o, args)
  args = args or {}
  o.populationSize = args.populationSize or 100
  o.eliteWeight = args.eliteWeight or 1.0
  o.crossoverWeight = args.crossoverWeight or 0.5
  o.mutantWeight = args.mutantWeight or 3.0
  o.noveltyWeight = args.noveltyWeight or 0.2
  return o
end)

GAEngine = class(function(o, args)
  args = args or {}
  o.options = GAOptions(args.options)
  o.scoringFunction = args.scoringFunction or nil
  o.spawnFunction = args.spawnFunction or nil
  o.mutationFunction = args.mutationFunction or nil
  o.crossoverFunction = args.crossoverFunction or nil
  o.population = shallowCopyArray(args.population)
  if #o.population > 0 and not o.population[1]:is_a(GAIndividual) then
    -- convert these things into GAIndividuals
    local tmpArr = {}
    for _, thing in ipairs(o.population) do
      insert(tmpArr, GAIndividual{thing=thing})
    end
    o.population = tmpArr
  end
  o.randomGen = args.randomGen or RealRandomGen()
  return o
end)

function GAEngine:populationSortedByScore()
  local spop = shallowCopyArray(self.population)
  table.sort(spop, function(left, right)
      return (left:latestScore() or -1.0) < (right:latestScore() or -1.0)
    end
  )
  return spop
end

function GAEngine:currentBestThing()
  if #self.population <= 0 then return nil end
  local spop = self:populationSortedByScore()
  return spop[#spop].thing
end

function GAEngine:iterate()
  local nextPopulation = {}
  local spop = self:populationSortedByScore()
  local eliteWeight = self.options.eliteWeight
  local crossoverWeight = self.options.crossoverWeight
  if #self.population <= 1 or not self.crossoverFunction then
    crossoverWeight = 0
    end
  local mutantWeight = self.options.mutantWeight
  if #self.population <= 0 then
    mutantWeight = 0
  end
  local noveltyWeight = self.options.noveltyWeight
  local totalWeight = eliteWeight + crossoverWeight + mutantWeight + noveltyWeight
  
  -- Attempt to create a number of mutants based on desired population size
  -- and relative weights. If a mutant is invalid (score < 0), it will not be
  -- added to the next population. This means that there may be fewer mutants
  -- than desired.
  local numMutantsToMake = floor(self.options.populationSize * mutantWeight / totalWeight)
  for i = 1, numMutantsToMake do
    -- pick a random thing from our population
    local individual = self.randomGen:pick(self.population)
    if i == 1 then
      individual = spop[#spop] -- Guarantee that we at least pick the top guy once
    end
    local newThing = self.mutationFunction(individual.thing, self.randomGen)
    local score = self.scoringFunction(newThing)
    if score >= 0 then
      individual = individual:clone()
      individual.thing = newThing
      insert(individual.scoreHistory, score)
      insert(nextPopulation, individual)
    end
  end
  
  -- Attempt to create a number of crossovers based on desired population size
  -- and relative weights. If a mutant is invalid (score < 0), it will not be
  -- added to the next population. This means that there may be fewer mutants
  -- than desired.
  local numCrossoversToMake = floor(self.options.populationSize * crossoverWeight / totalWeight)
  for i = 1, numCrossoversToMake do
    -- pick 2 random things from our population
    local individualA = self.population[self.randomGen:intRange(1, #self.population)]
    local individualB = self.population[self.randomGen:intRange(1, #self.population)]
    local newThing = self.crossoverFunction(individualA.thing, individualB.thing, self.randomGen)
    local score = self.scoringFunction(newThing)
    if score >= 0 then
      individualA = individualA:clone()
      individualA.thing = newThing
      insert(individualA.scoreHistory, score)
      insert(nextPopulation, individualA)
    end
  end

  -- Of the remaining space, insert our top elites based on ratio of elite & novelty weights.
  local numElites = min(
    floor(0.2 + (self.options.populationSize - #nextPopulation) * eliteWeight / (eliteWeight + noveltyWeight)),
    #spop)
  for i = #spop - numElites + 1, #spop do
    insert(nextPopulation, spop[i])
  end
  
  -- novel spawns fill the remainder. This will loop until populationSize is
  -- satisfied, so if `spawnFunction()` repeatedly returns invalid (score<0)
  -- things, it will loop forever.
  while #nextPopulation < self.options.populationSize do
    local newThing = self.spawnFunction(self.randomGen)
    local score = self.scoringFunction(newThing)
    if score >= 0 then
      insert(nextPopulation,
                   GAIndividual({thing = newThing, scoreHistory = {score}}))
    end
  end

  self.randomGen:shuffle(nextPopulation)
  self.population = nextPopulation
end

-- End genetic algorithm section

-- Begin Mat manipulation section

function getMat()
  for _, matGUID in ipairs(matSurfaceGUIDs) do
    local mat = getObjectFromGUID(matGUID)
    if mat then return mat end
  end
  return nil
end

function getMatSize()
  -- Default size if we don't have a mat
  local xInches = 60
  local zInches = 44

  local mat = getMat()
  if mat then
    -- 1.22, 1, .83 for 30x44 (combat patrol)
    -- 1.66, 1, 1.22 for 44x60 (strike force)
    -- 2.49, 1, 1.22 for 44x90 (onslaught)
    xInches = floor(0.5 + mat.getScale()[1] * 36.14)
    if xInches <= 44 then
      zInches = 30
    end
  end
  return {xInches=xInches, zInches=zInches}
end

function setMatImage(imageUrl)
  local mat = getMat()
  if not mat then return end
  local custom = mat.getCustomObject()
  custom.diffuse  = imageUrl
  mat.setCustomObject(custom)
  mat.reload()
end

-- End mat manipulation section

-- Info for an asset corresponding to a single TTS object. Currently only supports CustomModel.
AssetInfo = class(function(o, args)
  args = args or {}
  o.name = args.name or 'asset'
  -- Custom Model info
  o.custom = shallowCopyTable(args.custom)
  o.custom.material = o.custom.material or 3 -- cardboard
  o.custom.mesh = o.custom.mesh or nil
  o.custom.collider = o.custom.collider or nil
  o.custom.diffuse = o.custom.diffuse or nil
  o.tint = args.tint or nil -- e.g. Color(1, 0.5, 0.5)
  return o
end)

function AssetInfo:spawn(terrainFeature, terrainModel, isMirror, callback_function)
  -- copy data from the main object so that we get a decent starting point for what the data structure should look like.
  local featureTransform = terrainFeature.transform
  if isMirror then
    featureTransform = featureTransform:clone()
    featureTransform.posX = -featureTransform.posX
    featureTransform.posZ = -featureTransform.posZ
    featureTransform.rotY = (featureTransform.rotY + 180) % 360
  end
  local data = getObjectFromGUID(mainObjectGUID).getData()
  -- print(JSON.encode_pretty(data))
  data.Nickname = terrainFeature:getName()
  data.Description = terrainFeature:getDescription()
  data.GMNotes = ''
  data.CustomMesh.MeshURL = self.custom.mesh
  data.CustomMesh.ColliderURL = self.custom.collider or self.custom.mesh or ''
  data.CustomMesh.DiffuseURL = self.custom.diffuse or ''
  data.CustomMesh.NormalURL = self.custom.normal or ''
  data.CustomMesh.Convex = self.custom.isConvex or false
  data.Transform = terrainModel.transform:combine(featureTransform)
  data.Transform.posY = data.Transform.posY + matYOffset
  data.LuaScript = ''
  data.LuaScriptState = ''
  data.Locked = true
  data.Tooltip = true
  data.XmlUI = ''
  data.Tags = {spawnedTag}
  return spawnObjectData({data = data, callback_function = callback_function})
end

assets = {}
assets['cube'] = AssetInfo({
    name = 'cube',
    custom = {
      mesh = 'http://cloud-3.steamusercontent.com/ugc/1738926229610401119/52735B6291CE74406CF97147ACBC71B17C86DBF1/',
      collider = 'http://cloud-3.steamusercontent.com/ugc/1738926229610401119/52735B6291CE74406CF97147ACBC71B17C86DBF1/'
    }
  })

-- Strix's Containers
for _, r in ipairs({
    {'red', '1774951893767676952/714E436E805F70822CAF9FAF5391131B6975A4AD/'},
    {'green', '1774951893767645405/CADC3175DDBA44448F532219F103254B596C02BF/'},
    {'blue', '1774951893767646093/E8F186D7A7367609C56789686A7D5A365DEC19E3/'},
    {'blueorange', '1774951893767563018/747A974BE6E7940A17F8DFDEE02CB733D2B149DB/'}
    }) do
  assets['strix_container_' .. r[1]] =  AssetInfo({
    name='strix_container_' .. r[1],
    custom = {
      mesh = 'http://cloud-3.steamusercontent.com/ugc/1774951893767574211/0877B8330D7D3A2F63583614BC006963E936F867/',
      diffuse = 'http://cloud-3.steamusercontent.com/ugc/' .. r[2],
      collider = ''
    }
  })
end

-- Eddie manufactorum pieces
for _, r in ipairs({
    {'flattiles', '1675864420940323810/4A478336CD213B5FA50AFE7525C5305FAB9BCD50/'},
    {'wallflat', '1675864420940292014/CFBB9C4C0E597DF3F3FE5771865F9A3A2D2E77CA/'},
    {'chimney', '1675864420940300075/23854CAF71509AB159EDB088C455C4DCD292E03F/'},
    {'window', '1675864420940305805/7C8CC85B880B0ACF0C1DAD4D7E1A8836D8511988/'},
    {'opendoor', '1675864420940309615/48FE68E454826D7021CAC44678176BC9A46A3A40/'},
    {'closeddoor', '1675864420940295921/C27FF631C81C103040B97C16DD1371BACFE368ED/'},
    {'mechlogo', '1675864420940313814/ADD16B764F6380EDF3BFC279AA7AC545100C9BCF/'},
    {'walldynamo', '1675864420940288429/224322ECC7DA0376093F794C57697A98B96A020D/'},
    {'windowBrokenR', '1675864420940260504/13D58E85D41EACEFD63BF1328653DEEDF999001F/'},
    {'windowBrokenL', '1675864420940280717/B60A677639997A0E3DF54053827D5ED915D88F67/'},
    {'indenttiles', '1675864420940321586/CF439D7E3689A31CE90247AA0396AB9E565F3BE2/'},
    {'brokentiles', '1675864420940336622/DAE42228ECDA4A1D049D8C242EB33B46996B0EE9/'},
    {'railing', '1675864420940276292/D0690242953CEF0241E431D9164A9A769E5FB0C9/'},
    {'railingBroken', '1675864420940317515/4F1DAEBC722641137911702AA371F7ED9C1EB4EC/'},
    }) do
  assets['eddie_manufactorum_' .. r[1]] = AssetInfo({
    name='eddie_manufactorum_' .. r[1],
    custom = {
      mesh = 'http://cloud-3.steamusercontent.com/ugc/' .. r[2],
      collider = 'http://cloud-3.steamusercontent.com/ugc/' .. r[2],
      diffuse = 'http://cloud-3.steamusercontent.com/ugc/1675864420939890855/37BA5485CDC008949ED7525B31F43CFD3D146023/',
      normal = 'http://cloud-3.steamusercontent.com/ugc/1675864420939893488/7DF822A09523D89426B4777D0C24F525A910F57D/',
      isConvex = true
    }
  })
end






-- TerrainModel: info used to position an asset for use in constructing the visuals / physics in TTS
TerrainModel = class(function(o, args)
  args = args or {}
  -- position and rotation, relative to the center of a TerrainFeature whose position is right on the surface of the mat
  o.transform = Transform(args.transform)
  o.assetName = args.assetName or 'asset'
  -- if non-nil, only have this present in those state numbers. But multi-state terrain will be an advanced feature added in the far future.
  o.statesIncluded = args.statesIncluded or nil
  return o
end)

function TerrainModel:clone()
  return TerrainModel(self)
end

function cubeModel(o)
  local m = TerrainModel(o)
  m.assetName = 'cube'
  m.transform = m.transform:combine({posY = 1})
  return m
end

function strixContainerModel(o)
  local m = TerrainModel(o)
  m.assetName = 'strix_container_green'
  -- Scale to 5x2.5x2.5, and shift upward such that it sits on the ground
  m.transform = Transform(
    {posY=1.25, scaleX=0.82, scaleY=0.86, scaleZ=.79}):combine(m.transform)
  return m
end

function eddieManufactorumModel(nameOrNumber, o)
  -- figure out which we need
  local name = nameOrNumber
  if type(nameOrNumber) == "number" then
    name = ({'windowBrokenL', 'windowBrokenR', 'window', 'opendoor', 'closeddoor', 'wallflat', 'mechlogo', 'walldynamo', 'chimney'})[nameOrNumber]
  end
  
  -- Get the right offset
  -- Be -.05" down, as we want these ruins to be sunk a bit into the floor to be able to display objective disks through them.
  local offset = Transform({posY=-.06})
  -- most wall-type-things are offset a bit, and need 180 degree rotation.
  if arrayContains({'windowBrokenL', 'window', 'opendoor', 'closeddoor', 'wallflat', 'mechlogo', 'walldynamo', 'chimney'}, name) then
    offset.posZ = 1
    offset.rotY = 180
  end
  -- broken window with right sill, and both railings, have their origin in the bottom center, not its side center.
  if arrayContains({'windowBrokenR', 'railing', 'railingBroken'}, name) then
    offset.rotY = 180
  end
  o = shallowCopyTable(o)
  o.transform = offset:combine(o.transform)
  local m = TerrainModel(o)
  m.assetName = 'eddie_manufactorum_' .. name
  return m
end


-- An abstract rectangle, used for checking map validity and/or goodness
TerrainAbstraction = class(function(o, args)
  args = args or {}
  -- position and rotation and etc, relative to the center of a TerrainFeature whose position is right on the surface of the mat. This uses scaleX & scaleZ to represent its size in inches.
  o.transform = Transform(args.transform)
  o.isMirror = args.isMirror or false
  o.blocksLoS = args.blocksLoS or false
  o.obscuring = args.obscuring or false -- blocks LoS if only if you're not in it, kinda
  o.blocksMovement = args.blocksMovement or false
  o.isInternal = args.isInternal or false
  return o
end)

function TerrainAbstraction:clone()
  return TerrainAbstraction(self)
end
function TerrainAbstraction:distanceToBounds(myFeature, matX, matZ)
  return rectDistanceToBounds(self.transform:combine(myFeature.transform), matX, matZ)
end
function TerrainAbstraction:distanceTo(myTransform, otherAbstraction, otherTransform)
  return rectDistance(
    self.transform:combine(myTransform),
    otherAbstraction.transform:combine(otherTransform))
end

terrainFeatureClassDB = {}

TerrainFeature = class(function(o, args)
  args = args or {}
  o.featureClassName = nil
  -- Position of the terrain feature relative to the center surface of the mat. In normal operation, this should only utilize `posX`, `posZ`, `scale*`, and `rotY`.
  o.transform = Transform(args.transform)
  o.isCenterpiece = args.isCenterpiece or false
  return o
end)
function TerrainFeature:clone()
  return (terrainFeatureClassDB[self.featureClassName] or TerrainFeature)(self)
end
function deserializeTerrainFeature(o)
  return terrainFeatureClassDB[o.featureClassName or 'Cube'](o)
end
function TerrainFeature:getTerrainAbstractions() error('Unimplemented TerrainFeature:getTerrainAbstractions') end
function TerrainFeature:getTerrainModels() error('Unimplemented TerrainFeature:getTerrainModels') end
function TerrainFeature:getName() error('Unimplemented TerrainFeature:getName') end
function TerrainFeature:getDescription() error('Unimplemented TerrainFeature:getDescription') end
function TerrainFeature:getFlavor() return 'flavorless' end
function TerrainFeature:getApproxRadius() return error('Unimplemented TerrainFeature:getApproxRadius') end
function TerrainFeature:getApproxMoveBlockRadius() return self:getApproxRadius() end

function TerrainFeature:isEncroachingBoundary(matX, matZ, minGap, minMoveGap)
  -- If our approximate radius would not encroach, don't bother calculating the details; just return false.
  if (abs(self.transform.posX) + self:getApproxRadius() < matX * 0.5 - max(minGap, minMoveGap) and
      abs(self.transform.posZ) + self:getApproxRadius() < matZ * 0.5 - max(minGap, minMoveGap)) then
    return false
  end
  -- Our radius might encroach, so we have to check the details.
  for _, a in ipairs(self:getTerrainAbstractions()) do
    local dist = a:distanceToBounds(self, matX, matZ)
    if dist <= minGap then return true end
    if a.blocksMovement and dist <= minMoveGap then return true end
  end
  return false
end
function TerrainFeature:isEncroachingFeature(otherFeature, minGap, minMoveGap, isCheckingMirror)
  local myTransform = self.transform
  local otherTransform = otherFeature.transform
  if isCheckingMirror then
    otherTransform = otherTransform:clone()
    otherTransform.posX = -otherTransform.posX
    otherTransform.posZ = -otherTransform.posZ
    otherTransform.rotY = (otherTransform.rotY + 180) % 360
  end
  -- If our approximate radii would not encroach, don't bother calculating the details; just return false.
  local centerDistSquared = dist2DSquared(
    myTransform.posX, myTransform.posZ,
    otherTransform.posX, otherTransform.posZ)
  local minDist = self:getApproxRadius() + max(minGap, minMoveGap) + otherFeature:getApproxRadius()
  if minDist * minDist < centerDistSquared then
    return false
  end
  -- Our radii might encroach, so we have to check the details.
  for _, a in ipairs(self:getTerrainAbstractions()) do
    for _, b in ipairs(otherFeature:getTerrainAbstractions()) do
      local gapCheck = not (a.isInternal or b.isInternal)
      local moveCheck = a.blocksMovement and b.blocksMovement
      if gapCheck or moveCheck then
        local dist = a:distanceTo(myTransform, b, otherTransform)
        if gapCheck and dist <= minGap then return true end
        if moveCheck and dist <= minMoveGap then return true end
      end
    end
  end
  return false
end

-- Return either self (if nothing changed) or a new TerrainFeature of some kind
function TerrainFeature:mutated(map, randomGen)
  return self:clone()
end

CubeFeature = class(TerrainFeature, function(o, args)
  TerrainFeature.init(o, args)
  o.featureClassName = 'Cube'
  return o
end)
terrainFeatureClassDB['Cube'] = CubeFeature
function CubeFeature:getTerrainAbstractions() return {} end
function CubeFeature:getTerrainModels() return {cubeModel()} end
function CubeFeature:getName() return 'Cube' end
function CubeFeature:getDescription() return '' end

ContainerFeature = class(TerrainFeature, function(o, args)
  TerrainFeature.init(o, args)
  o.featureClassName = 'StrixContainers'
  return o
end)
terrainFeatureClassDB['StrixContainers'] = ContainerFeature
soloContainerAbstractions = {TerrainAbstraction({
      transform = {scaleX = 2.5, scaleZ = 5.0},
      blocksLoS = true,
      blocksMovement = true
  })}
function ContainerFeature:getTerrainAbstractions()
  return soloContainerAbstractions
end
function ContainerFeature:getTerrainModels()
  return {strixContainerModel(), strixContainerModel({transform={posY=2.5}})}
end
function ContainerFeature:getName() return 'Armoured Containers' end
function ContainerFeature:getDescription()
  return '[sup]2.5" height/width, 5" length. Can\'t see through tiny gaps'
end
function ContainerFeature:getFlavor() return 'containers' end
-- TODO: fix getApproxRadius by scale?
function ContainerFeature:getApproxRadius() return 2.8 end
function ContainerFeature:getApproxMoveBlockRadius() return 2.8 end

function ContainerFeature:mutated(map, randomGen)
  local containerConstructors = {
    LContainerFeature, JContainerFeature, IContainerFeature
  }
  return randomGen:pick(containerConstructors)(self)
end

LContainerFeature = class(ContainerFeature, function(o, args)
  TerrainFeature.init(o, args)
  o.featureClassName = 'StrixLContainers'
  return o
end)
terrainFeatureClassDB['StrixLContainers'] = LContainerFeature
LContainerAbstractions = {
  TerrainAbstraction({
      transform = {scaleX = 2.5, scaleZ = 5.0},
      blocksLoS = true,
      blocksMovement = true
    }),
  TerrainAbstraction({
      transform = {posX = 1.25, posZ = -3.75, scaleX = 5.0, scaleZ = 2.5},
      blocksLoS = true,
      blocksMovement = true
    })
}
function LContainerFeature:getTerrainAbstractions()
  return LContainerAbstractions
end
function LContainerFeature:getTerrainModels()
  return {
    strixContainerModel(),
    strixContainerModel({transform={posY=2.5}}),
    strixContainerModel({transform={posX = 1.25, posZ = -3.75, rotY = 90}}),
    strixContainerModel({transform={posX = 1.25, posZ = -3.75, rotY = 90, posY=2.5}}),
  }
end
function LContainerFeature:getApproxRadius() return 6.25 end
function LContainerFeature:getApproxMoveBlockRadius() return 6.25 end

JContainerFeature = class(ContainerFeature, function(o, args)
  TerrainFeature.init(o, args)
  o.featureClassName = 'StrixJContainers'
  return o
end)
terrainFeatureClassDB['StrixJContainers'] = JContainerFeature
JContainerAbstractions = {
  TerrainAbstraction({
      transform = {scaleX = 2.5, scaleZ = 5.0},
      blocksLoS = true,
      blocksMovement = true
    }),
  TerrainAbstraction({
      transform = {posX = -1.25, posZ = -3.75, scaleX = 5.0, scaleZ = 2.5},
      blocksLoS = true,
      blocksMovement = true
    })
}
function JContainerFeature:getTerrainAbstractions()
  return JContainerAbstractions
end
function JContainerFeature:getTerrainModels()
  return {
    strixContainerModel(),
    strixContainerModel({transform={posY=2.5}}),
    strixContainerModel({transform={posX = -1.25, posZ = -3.75, rotY = 90}}),
    strixContainerModel({transform={posX = -1.25, posZ = -3.75, rotY = 90, posY=2.5}}),
  }
end
function JContainerFeature:getApproxRadius() return 6.25 end
function JContainerFeature:getApproxMoveBlockRadius() return 6.25 end

IContainerFeature = class(ContainerFeature, function(o, args)
  TerrainFeature.init(o, args)
  o.featureClassName = 'StrixIContainers'
  return o
end)
terrainFeatureClassDB['StrixIContainers'] = IContainerFeature
IContainerAbstractions = {
  TerrainAbstraction({
      transform = {scaleX = 2.5, scaleZ = 5.0},
      blocksLoS = true,
      blocksMovement = true
    }),
  TerrainAbstraction({
      transform = {posZ = -5, scaleX = 2.5, scaleZ = 5.0},
      blocksLoS = true,
      blocksMovement = true
    })
}
function IContainerFeature:getTerrainAbstractions()
  return IContainerAbstractions
end
function IContainerFeature:getTerrainModels()
  return {
    strixContainerModel(),
    strixContainerModel({transform={posY=2.5}}),
    strixContainerModel({transform={posZ = -5}}),
    strixContainerModel({transform={posZ = -5, posY=2.5}}),
  }
end
function IContainerFeature:getApproxRadius() return 7.61 end
function IContainerFeature:getApproxMoveBlockRadius() return 7.61 end

-- End containers

-- Begin Ruin

--[[
Planned Ruin featureset?
* Rectangle ruin floors (done)
* Ruin walls, extending from corners, possibly inset an inch
* Higher floors (if it's the floor just above the ground, it also is considered to block movement)
* Windows/doors
* Multi-state ruins if multi-floor
--]]
RuinFeature = class(TerrainFeature, function(o, args)
  if not args.transform then
    -- default scale for 3" between levels with Eddie's manufactorum tileset
    -- args.transform = {scaleX=0.89, scaleY=0.89, scaleZ=0.89}
  end
  TerrainFeature.init(o, args)
  o.featureClassName = 'EddieManufactorumRuin'
  local randomGen = args.randomGen or RealRandomGen()
  -- Length (X) and Width (Z) in tiles (note that each tile is 2" wide). Always at least as long as wide WLOG.
  o.length = args.length or randomGen:intRange(3, 7)
  o.width = args.width or randomGen:intRange(2, o.length)
  o.isWallInset = args.isWallInset
  if o.isWallInset == nil then
    o.isWallInset = false
    if o.length > 2 and o.width > 2 then
      o.isWallInset = (randomGen:intRange(1, 2) == 2)
    end
  end
  o.isGroundWallOpaque = args.isGroundWallOpaque
  if o.isGroundWallOpaque == nil then
    o.isGroundWallOpaque = (o.length * o.width > 10) or (randomGen:intRange(1, 4) > 1)
  end
  o.isUpperWallOpaque = args.isUpperWallOpaque
  if o.isUpperWallOpaque == nil then
    o.isUpperWallOpaque = (o.length * o.width > 16) or o.isGroundWallOpaque and (randomGen:intRange(1, 3) > 1)
  end
  --[[ wallinfos
  an array of arrays of numbers, from ground floor up, representing each wall tile going clockwise starting with the +X +Z corner.
  Number meaning in manufactorum style:
  * 0: empty space
  * 1: damaged window, with frame on left (viewed from exterior)
  * 2: damaged window, with frame on right (viewed from exterior)
  * 3: intact window
  * 4: open door
  * 5: closed door
  * 6: wall, flat
  * 7: wall, with mechanicus logo
  * 8: wall, with dynamo extrusion
  * 9: wall, with chimney
  Note: 1+ blocks movement, 7+ blocks LoS too
  --]]
  o.wallInfos = args.wallInfos
  if not o.wallInfos then
    o.wallInfos = {}
    local wallLenX = o.length - (o.isWallInset and 1 or 0)
    local wallLenZ = o.width - (o.isWallInset and 1 or 0)
    local wallInfoLen = wallLenX * 2 + wallLenZ * 2
    -- corners: Always have a north-side (+Z) corner, sometimes both, and more rarely south-side corners
    local corners = {
      randomGen:intRange(1, 2) == 1, false,
      randomGen:intRange(1, 6) == 1, randomGen:intRange(1, 6) == 1}
    corners[2] = not corners[1]
    if wallLenZ > 3 and randomGen:intRange(1, 3) == 1 then
      corners[1] = true
      corners[2] = true
    end
    if wallLenZ < 4 then
      corners[3] = false
      corners[4] = false
    end
    -- prefer height 2, but sometimes other heights
    local possibleHeights = {2, 2, 2, 2, 3, 3}
    local height = possibleHeights[randomGen:intRange(1, #possibleHeights)]
    for h = 1, height do
      local wallInfo = {}
      for ix = 1, wallInfoLen do
        insert(wallInfo, 0)
      end
      local infoOffset = 0
      for cornerIx = 1, 4 do
        if corners[cornerIx] then
          local maxDistLeft = ({wallLenZ, wallLenX})[cornerIx % 2 + 1]
          local maxDistRight = ({wallLenX, wallLenZ})[cornerIx % 2 + 1]
          local distLeft = maxDistLeft
          local distRight = maxDistRight
          if h == 1 then
            distLeft = randomGen:intRange(min(2, maxDistLeft), max(2, maxDistLeft))
            distRight = randomGen:intRange(min(2, maxDistRight), max(2, maxDistRight))
          end
          for eIx = 0, distRight - 1 do
            local ix = (eIx + infoOffset) % wallInfoLen + 1
            -- if we're on the ground floor or there's something solid beneath us, we can place.
            if h == 1 or o.wallInfos[h-1][ix] >= 3 then
              wallInfo[ix] = 6
            end
          end
          for eIx = 0, distLeft - 1 do
            local ix = (-eIx + infoOffset - 1) % wallInfoLen + 1
            if h == 1 or o.wallInfos[h-1][ix] >= 3 then
              wallInfo[ix] = 6
            end
          end
          infoOffset = infoOffset + maxDistRight
        end
      end -- end corner
      
      -- TODO: add a check here to make sure the ruin is not totally enclosed. If there isn't a gap at least such-and-such (4?) long, add one.
      
      -- convert walls into other things
      for ix, item in ipairs(wallInfo) do
        local prev = wallInfo[(ix - 2) % wallInfoLen + 1]
        local later = wallInfo[ix % wallInfoLen + 1]
        if item > 0 then
          if h == 1 and o.isGroundWallOpaque then
            wallInfo[ix] = 6  -- wall
            if prev ~= 5 and later ~= 5 and randomGen:intRange(1, 4) == 1 then
              wallInfo[ix] = 5 -- sometimes a door
            end
          elseif h == 2 and o.isUpperWallOpaque then
            wallInfo[ix] = randomGen:intRange(6, 8)
          elseif h == 3 and o.isUpperWallOpaque then
            wallInfo[ix] = randomGen:intRange(6, 9)
          else -- non-opaque scenario
            wallInfo[ix] = 3 -- window
            if h == 1 and prev ~= 4 and later ~= 4 and randomGen:intRange(1, 4) == 1 then
              wallInfo[ix] = 4 -- sometimes open door on ground floor, nonconsecutively
            end
            if prev == 0 and later >= 3 and randomGen:intRange(1, 3) > 1 then
              wallInfo[ix] = 1 -- broken window, frame on left
            elseif prev >= 3 and later == 0 and randomGen:intRange(1, 3) > 1 then
              wallInfo[ix] = 2 -- broken window, frame on right
            end
          end
        end
      end -- end converting walls to other things
      insert(o.wallInfos, wallInfo)
    end
  end
  
  return o
end)
terrainFeatureClassDB['EddieManufactorumRuin'] = RuinFeature
function RuinFeature:getTerrainAbstractions()
  -- start with just the floor
  local l = {TerrainAbstraction({
        transform = {scaleX = self.length * 2, scaleZ = self.width * 2},
        obscuring = true
  })}
  if not self.wallInfos then return l end

  -- TODO: maybe clean this up somehow XD
  local groundWallInfo = self.wallInfos[1]
  local wallLenX = self.length - (self.isWallInset and 1 or 0)
  local wallLenZ = self.width - (self.isWallInset and 1 or 0)
  local xOffset = self.length - (self.isWallInset and 1 or 0)
  local zOffset = self.width - (self.isWallInset and 1 or 0)
  -- Note: 1+ blocks movement, 7+ blocks LoS too
  local wallInfoIx = 1
  local amInWall = false
  -- north wall
  for wz = 1, wallLenZ do
    if groundWallInfo[wallInfoIx] >= 1 then
      if not amInWall then
        amInWall = true
        insert(l, TerrainAbstraction({
          transform = {scaleX = 0.1, scaleZ = 2, posX=xOffset, posZ=zOffset + 1 - wz * 2},
          blocksLoS = self.isGroundWallOpaque,
          blocksMovement = true,
          isInternal = true
        }))
      else
        local t = l[#l].transform
        t.scaleZ = t.scaleZ + 2
        t.posZ = t.posZ - 1
      end
    else
      amInWall = false
    end
    wallInfoIx = wallInfoIx + 1
  end
  
  -- east wall
  amInWall = false
  for wx = 1, wallLenX do
    if groundWallInfo[wallInfoIx] >= 1 then
      if not amInWall then
        amInWall = true
        insert(l, TerrainAbstraction({
          transform = {scaleZ = 0.1, scaleX = 2, posZ=-zOffset, posX=xOffset + 1 - wx * 2},
          blocksLoS = self.isGroundWallOpaque,
          blocksMovement = true,
          isInternal = true
        }))
      else
        local t = l[#l].transform
        t.scaleX = t.scaleX + 2
        t.posX = t.posX - 1
      end
    else
      amInWall = false
    end
    wallInfoIx = wallInfoIx + 1
  end
  
  -- south wall
  amInWall = false
  for wz = 1, wallLenZ do
    if groundWallInfo[wallInfoIx] >= 1 then
      if not amInWall then
        amInWall = true
        insert(l, TerrainAbstraction({
          transform = {scaleX = 0.1, scaleZ = 2, posX=-xOffset, posZ=wz * 2 - 1 - zOffset},
          blocksLoS = self.isGroundWallOpaque,
          blocksMovement = true,
          isInternal = true
        }))
      else
        local t = l[#l].transform
        t.scaleZ = t.scaleZ + 2
        t.posZ = t.posZ + 1
      end
    else
      amInWall = false
    end
    wallInfoIx = wallInfoIx + 1
  end
    
  -- west wall
  amInWall = false
  for wx = 1, wallLenX do
    if groundWallInfo[wallInfoIx] >= 1 then
      if not amInWall then
        amInWall = true
        insert(l, TerrainAbstraction({
          transform = {scaleZ = 0.1, scaleX = 2, posZ=zOffset, posX=wx * 2 - 1 - xOffset},
          blocksLoS = self.isGroundWallOpaque,
          blocksMovement = true,
          isInternal = true
        }))
      else
        local t = l[#l].transform
        t.scaleX = t.scaleX + 2
        t.posX = t.posX + 1
      end
    else
      amInWall = false
    end
    wallInfoIx = wallInfoIx + 1
  end

  return l
end
function RuinFeature:getTerrainModels()
  local l = {}
  -- floor
  for tx = 1, self.length do
    for tz = 1, self.width do
      local x = tx * 2 - 1 - self.length
      local z = tz * 2 - 1 - self.width
      local rotY = (90 * RealRandomGen():intRange(1, 4)) % 360
      insert(l, eddieManufactorumModel('flattiles', {
            transform={posX=x, posZ=z, rotY = rotY}}))
    end
  end

  -- walls
  -- TODO: maybe clean this up somehow XD
  for h, wallInfo in ipairs(self.wallInfos or {}) do
    local wallLenX = self.length - (self.isWallInset and 1 or 0)
    local wallLenZ = self.width - (self.isWallInset and 1 or 0)
    local xOffset = self.length - (self.isWallInset and 1 or 0)
    local zOffset = self.width - (self.isWallInset and 1 or 0)
    local wallInfoIx = 1
    local posY = (h - 1) * 3.38
    -- north wall
    for wz = 1, wallLenZ do
      if wallInfo[wallInfoIx] > 0 then
        insert(l, eddieManufactorumModel(wallInfo[wallInfoIx], {transform={
              posX=xOffset,
              posZ=zOffset + 1 - wz * 2,
              posY=posY,
              rotY=0}}))
      end
      wallInfoIx = wallInfoIx + 1
    end
    -- east wall
    for wx = 1, wallLenX do
      if wallInfo[wallInfoIx] > 0 then
        insert(l, eddieManufactorumModel(wallInfo[wallInfoIx], {transform={
              posX=xOffset + 1 - wx * 2,
              posZ=-zOffset,
              posY=posY,
              rotY=90}}))
      end
      wallInfoIx = wallInfoIx + 1
    end
    -- south wall
    for wz = 1, wallLenZ do
      if wallInfo[wallInfoIx] > 0 then
        insert(l, eddieManufactorumModel(wallInfo[wallInfoIx], {transform={
              posX=-xOffset,
              posZ=wz * 2 - 1 - zOffset,
              posY=posY,
              rotY=180}}))
      end
      wallInfoIx = wallInfoIx + 1
    end
    -- west wall
    for wx = 1, wallLenX do
      if wallInfo[wallInfoIx] > 0 then
        insert(l, eddieManufactorumModel(wallInfo[wallInfoIx], {transform={
              posX=wx * 2 - 1 - xOffset,
              posZ=zOffset,
              posY=posY,
              rotY=270}}))
      end
      wallInfoIx = wallInfoIx + 1
    end
  end
  return l
end
function RuinFeature:getName() return 'Ruin' end
function RuinFeature:getDescription() return '' end
function RuinFeature:getFlavor() return 'ruins' end
function RuinFeature:getApproxRadius()
  -- TODO: fix this by scale?
  -- note: after a little testing, sqrt is fast enough that caching probably wouldn't matter
  return sqrt(self.width * self.width + self.length * self.length)
end
function RuinFeature:mutated(map, randomGen)
  return RuinFeature{randomGen=randomGen, transform=self.transform:clone()}
end

-- End Ruin

MapOptions = class(function(o, args)
  args = args or {}
  o.isRotationallySymmetric = args.isRotationallySymmetric or false
  o.xInches = args.xInches or 60
  o.zInches = args.zInches or 44
  o.matImage = args.matImage or 'https://steamuserimages-a.akamaihd.net/ugc/961990768477318654/C73F9B4F486C47CB2808138069F1DBBBE6B71B14/'
  o.minTerrainGap = args.minTerrainGap or 1.0
  o.minTerrainMoveGap = args.minTerrainMoveGap or 6.2
  o.minEdgeGap = args.minEdgeGap or 1.0
  o.minEdgeMoveGap = args.minEdgeMoveGap or 2.6
  o.minRuins = args.minRuins or 4
  o.maxRuins = args.maxRuins or 8
  o.minContainers = args.minContainers or 0
  o.maxContainers = args.maxContainers or 4
  return o
end)
function MapOptions:clone()
  return MapOptions(self)
end

Map = class(function(o, args, shallow)
  args = args or {}
  o.score = args.score
  if shallow then
    o.mapOptions = args.mapOptions
    o.terrainFeatures = shallowCopyArray(args.terrainFeatures)
  else
    o.mapOptions = MapOptions(args.mapOptions)
    o.terrainFeatures = {}
    for k, f in ipairs(args.terrainFeatures or {}) do
      insert(o.terrainFeatures, f:clone())
    end
  end
  return o
end)

function Map:clone()
  return Map(self)
end
function Map:cloneShallow()
  return Map(self, true)
end
function deserializeMap(m)
  m.terrainFeatures = shallowCopyArray(m.terrainFeatures)
  for i, f in ipairs(m.terrainFeatures) do
    m.terrainFeatures[i] = deserializeTerrainFeature(f)
  end
  return Map(m)
end
function Map:countFlavors()
  local flavorCounts = {containers=0, ruins=0, woods=0, flavorless=0}
  for _, f in ipairs(self.terrainFeatures) do
    local n = 1
    if self.mapOptions.isRotationallySymmetric and not f.isCenterpiece then
      n = 2
    end
    local flavor = f:getFlavor()
    flavorCounts[flavor] = flavorCounts[flavor] + n
  end
  return flavorCounts
end
function Map:isValid()
  -- todo? Or just assume each iterative mutation is ok?
  return true
end
function Map:getAllTerrainFeatures()
  -- todo: if I'm marked as rotationally mirrored, also generate copies in rotationally mirrored locations
  return self.terrainFeatures
end
function Map:wouldTerrainFeatureBeValid(terrainFeature, indexToIgnore)
  -- checking we're not too clouse to the map bounds
  if terrainFeature:isEncroachingBoundary(
      self.mapOptions.xInches, self.mapOptions.zInches,
      self.mapOptions.minEdgeGap, self.mapOptions.minEdgeMoveGap) then
    return false
  end

  -- if rotationally symmmetric, and not a centerpiece, check against the mirror of itself
  if (self.mapOptions.isRotationallySymmetric and not terrainFeature.isCenterpiece and
      terrainFeature:isEncroachingFeature(
        terrainFeature, self.mapOptions.minTerrainGap,
        self.mapOptions.minTerrainMoveGap, true)) then
    return false
  end
  
  -- check against all the extant terrain features. If rotationally symmetric, also check against the mirrors
  for i, f in ipairs(self.terrainFeatures) do
    if i ~= indexToIgnore then
      if terrainFeature:isEncroachingFeature(
          f, self.mapOptions.minTerrainGap, self.mapOptions.minTerrainMoveGap) then
        return false
      end
      if (self.mapOptions.isRotationallySymmetric and
          terrainFeature:isEncroachingFeature(
            f, self.mapOptions.minTerrainGap, self.mapOptions.minTerrainMoveGap, true)) then
        return false
      end
    end
  end
  return true
end

MapPopulation = class(function(o, args)
  args = args or {}
  o.timestamp = args.timestamp
  o.maps = {}
  for _, m in ipairs(args.maps or {}) do
    insert(o.maps, Map(m))
  end
  return o
end)

Carltographer = class(function(o, args)
  args = args or {}
  o.mapOptions = MapOptions(args.mapOptions)
  o.map = Map(args.map or {mapOptions=o.mapOptions})
  o.gaOptions = GAOptions(args.gaOptions or {
    populationSize=10,
    crossoverWeight=0.0, -- crossovers are just stupid expensive in the current implementation
  })
  o.numGeneticIterations = args.numGeneticIterations or 100
  o.selectedPopulationIndex = args.selectedPopulationIndex or nil
  o.populationHistory = {}
  for _, mp in ipairs(args.populationHistory or {}) do
    insert(o.populationHistory, MapPopulation(mp))
  end
  return o
end)
function deserializeCarltographer(c)
  if c.map then
    c.map = deserializeMap(c.map)
  end
  return Carltographer(c)
end

function scoreMap(map)
  if not map:isValid() then return -1 end
  
  -- Goodness drops off exponentially if you get out of the acceptable range of each terrain piece count.
  local flavorCounts = map:countFlavors()
  local containerGoodness = 2 ^ (-max(0,
      flavorCounts['containers'] - map.mapOptions.maxContainers,
      map.mapOptions.minContainers - flavorCounts['containers']))
  local ruinGoodness = 2 ^ (-max(0,
      flavorCounts['ruins'] - map.mapOptions.maxRuins,
      map.mapOptions.minRuins - flavorCounts['ruins']))
  
  -- more is better, i guess
  local countGoodness = 1.0 - 2^(- #map.terrainFeatures / 10.0)
  
  return containerGoodness * ruinGoodness * countGoodness
end

--[[
Random Map terrain piece modification attempts. These functions will attempt to do a thing to a terrain piece on a map, returning true if successful while modifying `map`, and returninf false if unsuccessful while leaving `map` unmodified.
--]]
function addRandomTerrainFeatureToMap(map, randomGen, constructors)
  local maxX = map.mapOptions.xInches / 2.0 - map.mapOptions.minEdgeMoveGap
  local maxZ = map.mapOptions.zInches / 2.0 - map.mapOptions.minEdgeMoveGap
  local x = randomGen:floatRange(-maxX, maxX)
  local z = randomGen:floatRange(-maxZ, maxZ)
  local rotY = 45 * randomGen:intRange(0, 7)
  local feature = randomGen:pick(constructors)(
    {randomGen=randomGen, transform={posX=x, posZ=z, rotY=rotY}})
  if map:wouldTerrainFeatureBeValid(feature, nil) then
    insert(map.terrainFeatures, feature)
    return true
  end
  return false
end

function modifyRandomTerrainFeatureInMap(map, randomGen)
  if #map.terrainFeatures <= 0 then
    return false
  end
  local terrainIx = randomGen:intRange(1, #map.terrainFeatures)
  local feature = map.terrainFeatures[terrainIx]:clone():mutated(map, randomGen)
  if map:wouldTerrainFeatureBeValid(feature, terrainIx) then
    map.terrainFeatures[terrainIx] = feature
    return true
  end
  return false -- true
end

function rotateRandomTerrainFeatureInMap(map, randomGen)
  if #map.terrainFeatures <= 0 then
    return false
  end
  local terrainIx = randomGen:intRange(1, #map.terrainFeatures)
  local feature = map.terrainFeatures[terrainIx]:clone()
  feature.transform.rotY = (
    feature.transform.rotY + randomGen:intRange(0, 1) * 90 - 45) % 360
  if map:wouldTerrainFeatureBeValid(feature, terrainIx) then
    map.terrainFeatures[terrainIx] = feature
    return true
  end
  return false
end

function moveRandomTerrainFeatureInMap(map, randomGen)
  if #map.terrainFeatures <= 0 then
    return false
  end
  local terrainIx = randomGen:intRange(1, #map.terrainFeatures)
  local feature = map.terrainFeatures[terrainIx]
  -- centerpieces don't move!
  if feature.isCenterpiece then
    return false
  end
  feature = feature:clone()
  -- todo: consider finding max distance to move based on closest terrain feature and desired gaps, and limiting vector so as not to encroach on table edge; this would reduce chances of an error. Or maybe just do these reductions if we find we would be encroaching.
  local dX = randomGen:floatRange(-1.0, 1.0)
  local dZ = randomGen:floatRange(-1.0, 1.0)
  feature.transform.posX = feature.transform.posX + dX
  feature.transform.posZ = feature.transform.posZ + dZ
  if map:wouldTerrainFeatureBeValid(feature, terrainIx) then
    map.terrainFeatures[terrainIx] = feature
    return true
  end
  return false
end

--[[
Attempts to remove a random terrain feature from `map`. If it fails to do so, returns `false` and leaves `map` unmodified. If successful, `map` will have one fewer terrain feature and the function returns `true`.
--]]
function removeRandomTerrainFeatureFromMap(map, randomGen, flavor)
  local indices = {}
  for ix, f in ipairs(map.terrainFeatures) do
    if f:getFlavor() == flavor then
      insert(indices, ix)
    end
  end
  if #indices < 1 then return false end
  table.remove(map.terrainFeatures, randomGen:pick(indices))
  return true
end

function mutateMap(map, randomGen)
  --[[ future thoughts:
  * have some weights on which action to try: add a terrain feature, mutate a terrain feature, rotate a terrain feature, move a terrain feature, remove a terrain feature
  * can weight things more on adding terrain if the quantity is low
  * validation check can be made nontrivially faster by only checking collision/encroachment on the particular feature being added or changed.
  * for movement of a terrain feature, can get a (hopefully?) safe move distance by checking distance to all other terrain features, and also bounding the movement vector by the battlefield edge encroachment.
  * should probably add "bounding box" transform to each terrain feature (one for full size, one for move blocking), which can be used to first check whether checking the details of the terrain feature's component abstractions is worthwhile. This can be optional, as some terrain features will be sufficiently simple that this isn't worth it.
    * actually, just a bounding ball might be simpler. Center + radius (determined from the most extreme point of its rectangles).
  --]]
  map = map:cloneShallow()
  local flavorCounts = map:countFlavors()
  local addContainerWeight = max(map.mapOptions.maxContainers - flavorCounts['containers'], 0)
  local addRuinWeight = max(map.mapOptions.maxRuins - flavorCounts['ruins'], 0)
  local removeContainerWeight = max(flavorCounts['containers'] - map.mapOptions.maxContainers, 0)
  local removeRuinWeight = max(flavorCounts['ruins'] - map.mapOptions.minRuins, 0)
  local moveWeight = 1
  local rotateWeight = 1
  local modifyWeight = 1
  if #map.terrainFeatures < 1 then
    moveWeight = 0
    rotateWeight = 0
    modifyWeight = 0
  end
  
  for _ = 1, 5 do -- 5 tries to get something to happen
    local pick = randomGen:weightedPick({
        {'addContainer', addContainerWeight},
        {'addRuin', addRuinWeight},
        {'removeContainer', removeContainerWeight},
        {'removeRuin', removeRuinWeight},
        {'move', moveWeight},
        {'rotate', rotateWeight},
        {'modify', modifyWeight}
      })
    if pick == 'addContainer' then
      if addRandomTerrainFeatureToMap(map, randomGen, {LContainerFeature, JContainerFeature, IContainerFeature}) then return map end
    elseif pick == 'addRuin' then
      if addRandomTerrainFeatureToMap(map, randomGen, {RuinFeature}) then return map end
    elseif pick == 'removeContainer' then
      if removeRandomTerrainFeatureFromMap(map, randomGen, 'containers') then return map end
    elseif pick == 'removeRuin' then
      if removeRandomTerrainFeatureFromMap(map, randomGen, 'ruins') then return map end
    elseif pick == 'move' then
      if moveRandomTerrainFeatureInMap(map, randomGen) then return map end
    elseif pick == 'rotate' then
      if rotateRandomTerrainFeatureInMap(map, randomGen) then return map end
    elseif pick == 'modify' then
     if modifyRandomTerrainFeatureInMap(map, randomGen) then return map end
    end
  end
  -- only get here if we failed all the above tries. Return a blank map as things were too hard.
  map.terrainFeatures = {}
  return map
end

function crossoverMaps(map1, map2, randomGen)
  local map = map1:clone()
  -- construct candidates as a list of copies of both maps' features.
  local terrainCandidates = map2:clone().terrainFeatures
  for _, feature in ipairs(map.terrainFeatures) do
    insert(terrainCandidates, feature)
  end
  randomGen:shuffle(terrainCandidates)
  -- clear `map` terrain features, and add from candidates if valid
  map.terrainFeatures = {}
  for _, feature in ipairs(terrainCandidates) do
    if map:wouldTerrainFeatureBeValid(feature, nil) then
      insert(map.terrainFeatures, feature)
    end
  end

  return map
end

function Carltographer:runGeneticAlgorithm(randomGen, isRefining, isContinuing)
  local timestamp = os.date("!%Y-%m-%dT%H:%M:%SZ")
  local pop = nil
  if isContinuing and self.selectedPopulationIndex and self.populationHistory[self.selectedPopulationIndex] then
    pop = {}
    for _, m in ipairs(self.populationHistory[self.selectedPopulationIndex].maps) do
      m = m:clone()
      m.mapOptions = self.mapOptions
      insert(pop, m)
    end
  end
  local ga = GAEngine({
      scoringFunction = scoreMap,
      spawnFunction = function()
        if isRefining then
          return self.map
        else
          return Map({mapOptions=self.mapOptions})
        end
      end,
      population = pop,
      mutationFunction = mutateMap,
      crossoverFunction = crossoverMaps,
      options = self.gaOptions
    })
  for i = 1, self.numGeneticIterations do
    ga:iterate()
  end
  local mapPopulation = MapPopulation({timestamp=timestamp})
  local spop = ga:populationSortedByScore()
  for i = #spop, 1, -1 do
    local map = spop[i].thing:clone()
    map.score = spop[i]:latestScore()
    insert(mapPopulation.maps, map)
  end
  insert(self.populationHistory, 1, mapPopulation)
  self.map = mapPopulation.maps[1]
  self.selectedPopulationIndex = 1
end

MAP_OPTIONS_NUMBER_TEXT_KEYS = {'minEdgeGap', 'minEdgeMoveGap', 'minTerrainGap', 'minTerrainMoveGap', 'minRuins', 'maxRuins', 'minContainers', 'maxContainers'}

function Carltographer:updateUI()
  local obj = getObjectFromGUID(mainObjectGUID)
  
  -- Buttons
  if getCurrentMapInput() then
    obj.UI.setAttribute('spawnButton', 'interactable', true)
    obj.UI.setAttribute('refineButton', 'interactable', true)
  else
    obj.UI.setAttribute('spawnButton', 'interactable', false)
    obj.UI.setAttribute('refineButton', 'interactable', false)
  end
  if self.selectedPopulationIndex and self.populationHistory[self.selectedPopulationIndex] then
    obj.UI.setAttribute('continuePopButton', 'interactable', true)
  else
    obj.UI.setAttribute('continuePopButton', 'interactable', false)
  end
  -- End buttons
  
  -- Map configs
  obj.UI.setAttribute('isRotationallySymmetric', "isOn",
    tostring(self.mapOptions.isRotationallySymmetric))
  
  for _, k in ipairs(MAP_OPTIONS_NUMBER_TEXT_KEYS) do
    obj.UI.setAttribute(k, 'text', tostring(self.mapOptions[k]))
  end
  -- End map configs
  
  -- population history
  for i=1, 10 do
    local id = 'populationButton' .. tostring(i)
    if i <= #self.populationHistory then
      obj.UI.setAttribute(id, 'text', self.populationHistory[i].timestamp)
      if self.selectedPopulationIndex == i then
        obj.UI.setAttribute(id, 'color', 'blue')
      else
        obj.UI.setAttribute(id, 'color', 'white')
      end
      obj.UI.setAttribute(id, 'interactable', true)
    else
      obj.UI.setAttribute(id, 'text', '')
      obj.UI.setAttribute(id, 'color', 'white')
      obj.UI.setAttribute(id, 'interactable', false)
    end
  end
  -- end population history
  
  -- population history
  for i=1, 10 do
    local id = 'individualButton' .. tostring(i)
    local ph = self.selectedPopulationIndex and self.populationHistory[self.selectedPopulationIndex]
    if ph and i <= #ph.maps then
      obj.UI.setAttribute(id, 'text', 'Map ' .. tostring(i))
      obj.UI.setAttribute(id, 'interactable', true)
    else
      obj.UI.setAttribute(id, 'text', '')
      obj.UI.setAttribute(id, 'interactable', false)
    end
  end
  -- end population history
end

function clickedPopulationButton(player, value, id)
  local ix = tonumber(string.match(id, "%d+"))
  print('clickedPopulationButton ', ix)
  local ct = centralCarltographer
  ct.selectedPopulationIndex = ix
  ct:updateUI()
end

function clickedIndividualButton(player, value, id)
  local ix = tonumber(string.match(id, "%d+"))
  local ct = centralCarltographer
  ct.map = ct.populationHistory[ct.selectedPopulationIndex].maps[ix]:clone()
  setCurrentMapInputStr(ct.map)
  ct:applyMapToMat()
  ct:updateUI()
end

function setCurrentMapInputStr(map)
  local obj = getObjectFromGUID(mainObjectGUID)
  map = map:cloneShallow()
  map.mapOptions = nil
  map.score = nil
  -- encoding map into text input. Note that we add whitespace to avoid some weird TTS object interpretation thing.
  local mapStr = ' ' .. (JSON.encode(map) or '') .. ' '
  obj.UI.setAttribute('currentMapInput', 'text', mapStr)
end

function getCurrentMapInput()
  local obj = getObjectFromGUID(mainObjectGUID)
  local mapStr = obj.UI.getAttribute('currentMapInput', 'text') or ''
  if not string.find(mapStr, '{') then
    return nil
  end
  local decodedMap = JSON.decode(mapStr)
  if decodedMap then
    return deserializeMap(decodedMap)
  end
  return nil
end

function Carltographer:updateFromUI()
  local obj = getObjectFromGUID(mainObjectGUID)
  self.mapOptions.isRotationallySymmetric = isToggleOn('isRotationallySymmetric')
  for _, k in ipairs(MAP_OPTIONS_NUMBER_TEXT_KEYS) do
    self.mapOptions[k] = tonumber(obj.UI.getAttribute(k, 'text') or '0')
  end
  self:updateUI()
end

function updateFromObjUI()
  startLuaCoroutine(self, 'updatefromUICoroutine')
end

function updatedTextInput(player, value, id)
  local obj = getObjectFromGUID(mainObjectGUID)
  obj.UI.setAttribute(id, "text", value)
  updateFromObjUI()
end

function updatedToggle(player, value, id)
  local obj = getObjectFromGUID(mainObjectGUID)
  obj.UI.setAttribute(id, "isOn", value)
  updateFromObjUI()
end

function updatefromUICoroutine()
  coroutine.yield(0)
  centralCarltographer:updateFromUI()
  return 1
end

function isToggleOn(id)
  local obj = getObjectFromGUID(mainObjectGUID)
  local x = obj.UI.getAttribute(id, 'isOn')
  if(not x or x == "False" or x == 'false' or x == nil) then
    return false
  else
    return true
  end
end


function Carltographer:applyMapToMat()
  for _, o in pairs(getObjectsWithTag(spawnedTag)) do
    destroyObject(o)
  end
  setMatImage(self.map.mapOptions.matImage)

  for _, f in pairs(self.map:getAllTerrainFeatures()) do
    local rootObj = nil
    local spawnTodos = 0
    local withMirrors = {false}
    if self.map.mapOptions.isRotationallySymmetric and not f.isCenterpiece then
      insert(withMirrors, true)
    end
    for _, isMirror in ipairs(withMirrors) do
      for _, m in ipairs(f:getTerrainModels()) do
        spawnTodos = spawnTodos + 1
        local index = spawnTodos -- 1-indexed
        local asset = assets[m.assetName]
        if not asset then
          JSON.encode_pretty(f:getTerrainModels())
          print('failed to find asset: ', JSON.encode_pretty(m))
        end
        asset:spawn(f, m, isMirror,
          function (spawned_object)
            spawnTodos = spawnTodos - 1
            if spawnTodos <= 0 then
              -- todo: everything got spawned, so time to join?
            end
          end
        )
        if not rootObj then
          rootObj = obj
        end
      end
    end
  end
end

function spawnMapFromTextInput()
  local ct = centralCarltographer
  local map = getCurrentMapInput()
  map.mapOptions = ct.mapOptions
  ct.map = map
  ct:applyMapToMat()
  updateFromObjUI()
end

function spawnFreshMap()
  print('creating random map')
  local ct = centralCarltographer
  local matSize = getMatSize()
  ct.mapOptions.xInches = matSize.xInches
  ct.mapOptions.zInches = matSize.zInches
  ct:runGeneticAlgorithm(RealRandomGen())
  print('created fresh random map')
  ct:applyMapToMat()
  setCurrentMapInputStr(ct.map)
  updateFromObjUI()
end

function continuePopulation()
  print('continuing population')
  local ct = centralCarltographer
  local matSize = getMatSize()
  ct.mapOptions.xInches = matSize.xInches
  ct.mapOptions.zInches = matSize.zInches
  ct:runGeneticAlgorithm(RealRandomGen(), false, true)
  print('continued population')
  ct:applyMapToMat()
  setCurrentMapInputStr(ct.map)
  updateFromObjUI()
end

function refineMap()
  print('refining map')
  local ct = centralCarltographer
  local matSize = getMatSize()
  ct.mapOptions.xInches = matSize.xInches
  ct.mapOptions.zInches = matSize.zInches
  ct.map.mapOptions = ct.mapOptions:clone()
  ct:runGeneticAlgorithm(RealRandomGen(), true)
  print('refined map')
  ct:applyMapToMat()
  setCurrentMapInputStr(ct.map)
  updateFromObjUI()
end

function updateUICoroutine()
  coroutine.yield(0)
  centralCarltographer:updateUI()
  return 1
end

function onLoad(stateString)
  centralCarltographer = deserializeCarltographer(JSON.decode(stateString) or {})
  -- update the UI in a coroutine because it seems like additive load and such requires a little time pause before the UI can be updated? :shrug:
  startLuaCoroutine(self, 'updateUICoroutine')
  print('Carltographer loaded')
end

function onSave()
  return JSON.encode(centralCarltographer or {})
end