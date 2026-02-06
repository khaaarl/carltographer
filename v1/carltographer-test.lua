require('carltographer')
local json = require('json')
local lu = require('luaunit')

MockRandomGen = class(RandomGen, function(o, args)
  args = args or {}
  o.intSequence = args.intSequence or {}
  return o
end)

function MockRandomGen:intRange(startInt, endInt)
  lu.assertTrue(#self.intSequence > 0)
  local i = self.intSequence[1]
  lu.assertTrue(i >= startInt)
  lu.assertTrue(i <= endInt)
  table.remove(self.intSequence, 1)
  return i
end

TestCarltographer = {} --class
    function TestCarltographer:testdist2DSquared()
      lu.assertEquals(dist2DSquared(1, 1, 1, 1), 0)
      lu.assertEquals(dist2DSquared(1, 1, 2, 1), 1)
      lu.assertEquals(dist2DSquared(1, -1, 2, 1), 5)
      lu.assertEquals(dist2DSquared(1, -1, 2.5, 1), 6.25)
    end
    
    function TestCarltographer:testdist2D()
      lu.assertEquals(dist2D(1, 1, 1, 1), 0)
      lu.assertEquals(dist2D(1, 1, 2, 1), 1)
      lu.assertEquals(dist2D(1, -1, 4, 3), 5)
      lu.assertEquals(dist2D(1, -1, 2.5, 1), 2.5)
    end
    
    function TestCarltographer:testdist2DSegmentPoint()
      lu.assertEquals(dist2DSegmentPoint(1, 1, 2, 1, 1.5, -1), 2)
      lu.assertEquals(dist2DSegmentPoint(1, 1, 2, 1, 5, 1), 3)
      lu.assertEquals(dist2DSegmentPoint(1, 1, 2, 1, -3, 1), 4)
      lu.assertEquals(dist2DSegmentPoint(0, 0, -7, 0, 3, 4), 5)
      lu.assertEquals(dist2DSegmentPoint(1, 1, 2, 1, 1.5, 1), 0)
      lu.assertEquals(dist2DSegmentPoint(1, 1, 2, 1, 2, 1), 0)
      lu.assertEquals(dist2DSegmentPoint(1, 1, 2, 1, 1, 1), 0)
      lu.assertEquals(dist2DSegmentPoint(1, 1, 1, 1, 1, 1), 0)
      lu.assertEquals(dist2DSegmentPoint(1, 1, 1, 1, 2, 1), 1)
    end
    
    function TestCarltographer:testcollide2DSegments()
      lu.assertFalse(collide2DSegments(0, 0, 1, 1, -1, 0, 0, 1))
      lu.assertTrue(collide2DSegments(0, 0, 1, 1, 0, 1, 1, 0))
      -- I'm sure there are way more cases that should be checked here heh
    end
    
    function TestCarltographer:testTransform()
      local t = Transform()
      lu.assertEquals(t.posX, 0)
      t.posX = 1
      local t2 = t:combine({rotY = 90})
      lu.assertEquals(t.posX, 1)
      lu.assertEquals(t.rotY, 0)
      lu.assertAlmostEquals(t2.posX, 0)
      lu.assertAlmostEquals(t2.posZ, -1)
      lu.assertEquals(t2.rotY, 90)
    end
    
    function TestCarltographer:testRectDistanceToBounds()
      local t = Transform()
      t.posX = 17
      t.scaleX = 6
      lu.assertAlmostEquals(rectDistanceToBounds(t, 44, 30), 2)
      t.posX = -17
      lu.assertAlmostEquals(rectDistanceToBounds(t, 44, 30), 2)
      t.posZ = 13
      t.scaleZ = 2
      lu.assertAlmostEquals(rectDistanceToBounds(t, 44, 30), 1)
      t.scaleX = 2
      t.rotY = 45
      lu.assertAlmostEquals(rectDistanceToBounds(t, 44, 30), 2 - math.sqrt(2), 0.00001)
      t.posZ = 15
      lu.assertAlmostEquals(rectDistanceToBounds(t, 44, 30), -math.sqrt(2), 0.00001)
    end
    
    function TestCarltographer:testRectDistance()
      local t1 = Transform()
      local t2 = Transform()
      t1.posX = 1
      t1.scaleX = 2
      t1.scaleZ = 2
      t2.posX = 4
      t2.scaleX = 2
      t2.scaleZ = 2
      lu.assertAlmostEquals(rectDistance(t1, t2), 1)
      t1.posX = 0
      t2.posX = 0
      t2.posZ = 6
      lu.assertAlmostEquals(rectDistance(t1, t2), 4)
      t2.posZ = 2
      lu.assertAlmostEquals(rectDistance(t1, t2), 0)
      t2.posZ = 6
      t2.rotY = 45
      lu.assertAlmostEquals(rectDistance(t1, t2), 5 - math.sqrt(2))
      t2.posZ = 0.5
      t2.posX = 0.5
      lu.assertAlmostEquals(rectDistance(t1, t2), 0)
      t2 = t1
      lu.assertAlmostEquals(rectDistance(t1, t2), 0)
    end
    
    function TestCarltographer:testGAIncrements()
      ga = GAEngine({
          scoringFunction = function(thing) return thing end,
          spawnFunction = function() return 0 end,
          mutationFunction = function(thing, randomGen)
            return thing + 1
          end,
          options = {populationSize = 3}
        })
  
      lu.assertEquals(#ga.population, 0)
      lu.assertNil(ga:currentBestThing())
  
      -- First iteration just spawns the population.
      ga:iterate()
      lu.assertEquals(#ga.population, 3)
      lu.assertNotNil(ga:currentBestThing())
      lu.assertEquals(ga:currentBestThing(), 0)
      
      -- Each gets bigger
      ga:iterate()
      lu.assertEquals(ga:currentBestThing(), 1)
      ga:iterate()
      lu.assertEquals(ga:currentBestThing(), 2)
      ga:iterate()
      lu.assertEquals(ga:currentBestThing(), 3)
      ga:iterate()
      lu.assertEquals(ga:currentBestThing(), 4)
      ga:iterate()
      lu.assertEquals(ga:currentBestThing(), 5)
      ga:iterate()
      lu.assertEquals(ga:currentBestThing(), 6)
      ga:iterate()
      lu.assertEquals(ga:currentBestThing(), 7)
      ga:iterate()
      lu.assertEquals(ga:currentBestThing(), 8)
    end
    
    function TestCarltographer:testMutateContainer()
      local f1 = JContainerFeature()
      local f2 = f1:mutated(Map(), MockRandomGen{intSequence={1}})
      lu.assertTrue(f2:is_a(LContainerFeature))
    end
    
    function TestCarltographer:testPositioningValidityAtEdge()
      local map = Map{mapOptions={xInches = 44, zInches = 30}}
      local feature = ContainerFeature()
      lu.assertTrue(map:wouldTerrainFeatureBeValid(feature))
      feature.transform.posX = 280
      lu.assertFalse(map:wouldTerrainFeatureBeValid(feature))
      feature.transform.posX = 21
      lu.assertFalse(map:wouldTerrainFeatureBeValid(feature))
      feature.transform.posX = 10
      lu.assertTrue(map:wouldTerrainFeatureBeValid(feature))
    end

    function TestCarltographer:testPositioningValidityVsOtherFeature()
      local map = Map({xInches = 44, zInches = 30})
      local feature = ContainerFeature()
      lu.assertTrue(map:wouldTerrainFeatureBeValid(feature))
      
      table.insert(map.terrainFeatures, feature:clone())
      lu.assertFalse(map:wouldTerrainFeatureBeValid(feature))
      feature.transform.posX = 1 -- not directly on top, but overlapping
      lu.assertFalse(map:wouldTerrainFeatureBeValid(feature))
      feature.transform.posX = 11 -- far enough to not encroach
      lu.assertTrue(map:wouldTerrainFeatureBeValid(feature))
      feature.transform.posX = 8 -- not overlapping, but encroaching
      lu.assertFalse(map:wouldTerrainFeatureBeValid(feature))
    end

    function TestCarltographer:testRuinWallsColliders()
      local ruinArgs = {
        length=3,
        width=3,
        isWallInset=false,
        wallInfos={{0,0,0,0,0,0,0,0,0,0,0,0}}
      }
      local ruin1 = RuinFeature(ruinArgs)
      -- just has floor
      lu.assertEquals(#ruin1:getTerrainAbstractions(), 1)
      lu.assertTrue(ruin1:getTerrainAbstractions()[1].obscuring)
      
      -- now let's add a partial wall
      ruinArgs.wallInfos = {{8,0,0,0,0,0,0,0,0,0,0,0}}
      local ruin2 = RuinFeature(ruinArgs)
      lu.assertEquals(#ruin2:getTerrainAbstractions(), 2)
      lu.assertTrue(ruin2:getTerrainAbstractions()[1].obscuring)
      local wallCollider = ruin2:getTerrainAbstractions()[2]
      lu.assertFalse(wallCollider.obscuring)
      lu.assertTrue(wallCollider.blocksMovement)
      lu.assertEquals(wallCollider.transform.posX, 3)
      lu.assertEquals(wallCollider.transform.posZ, 2)
      lu.assertEquals(wallCollider.transform.scaleX, 0.1)
      lu.assertEquals(wallCollider.transform.scaleZ, 2)
      
      -- how about a bigger wall; does concatenation work
      ruinArgs.wallInfos = {{8,8,0,0,0,0,0,0,0,0,0,0}}
      local ruin3 = RuinFeature(ruinArgs)
      lu.assertEquals(#ruin3:getTerrainAbstractions(), 2)
      lu.assertTrue(ruin3:getTerrainAbstractions()[1].obscuring)
      wallCollider = ruin3:getTerrainAbstractions()[2]
      lu.assertFalse(wallCollider.obscuring)
      lu.assertTrue(wallCollider.blocksMovement)
      lu.assertEquals(wallCollider.transform.posX, 3)
      lu.assertEquals(wallCollider.transform.posZ, 1)
      lu.assertEquals(wallCollider.transform.scaleX, 0.1)
      lu.assertEquals(wallCollider.transform.scaleZ, 4)
      
      -- a split wall
      ruinArgs.wallInfos = {{8,0,8,0,0,0,0,0,0,0,0,0}}
      local ruin4 = RuinFeature(ruinArgs)
      lu.assertEquals(#ruin4:getTerrainAbstractions(), 3)
      lu.assertTrue(ruin4:getTerrainAbstractions()[1].obscuring)
      local wallCollider1 = ruin4:getTerrainAbstractions()[2]
      local wallCollider2 = ruin4:getTerrainAbstractions()[3]
      lu.assertFalse(wallCollider1.obscuring)
      lu.assertTrue(wallCollider1.blocksMovement)
      lu.assertEquals(wallCollider1.transform.posX, 3)
      lu.assertEquals(wallCollider1.transform.posZ, 2)
      lu.assertEquals(wallCollider1.transform.scaleX, 0.1)
      lu.assertEquals(wallCollider1.transform.scaleZ, 2)
      lu.assertFalse(wallCollider2.obscuring)
      lu.assertTrue(wallCollider2.blocksMovement)
      lu.assertEquals(wallCollider2.transform.posX, 3)
      lu.assertEquals(wallCollider2.transform.posZ, -2)
      lu.assertEquals(wallCollider2.transform.scaleX, 0.1)
      lu.assertEquals(wallCollider2.transform.scaleZ, 2)
    end

    function TestCarltographer:testRuinWallsAndEncroaching()
      local map = Map({xInches = 60, zInches = 44, minTerrainGap=0.5, minTerrainMoveGap=6.2})
      local container = ContainerFeature()
      lu.assertTrue(map:wouldTerrainFeatureBeValid(container))
      table.insert(map.terrainFeatures, container:clone())
      
      -- ruin with no walls
      local ruinArgs = {
        width=2,
        length=2,
        isWallInset=false,
        transform={posX=-5},
        wallInfos={{0,0,0,0,0,0,0,0}}
      }
      local ruin1 = RuinFeature(ruinArgs)
      lu.assertEquals(#ruin1:getTerrainAbstractions(), 1)
      -- far enough that we're not encroaching the container
      lu.assertTrue(map:wouldTerrainFeatureBeValid(ruin1))
      
      -- now let's add a partial wall
      ruinArgs.wallInfos = {{8,0,0,0,0,0,0,0}}
      local ruin2 = RuinFeature(ruinArgs)
      lu.assertEquals(#ruin2:getTerrainAbstractions(), 2)
      -- now that we partly block movement, we're encroaching.
      lu.assertFalse(map:wouldTerrainFeatureBeValid(ruin2))
      
      -- clear out the container, and let's put in ruin2
      map.terrainFeatures = {ruin2:clone()}
      
      -- a new wall-less ruin
      ruinArgs.wallInfos = {{0,0,0,0,0,0,0,0}}
      ruinArgs.transform = {posZ = 7, posX = -5}
      local ruin3 = RuinFeature(ruinArgs)
      lu.assertEquals(#ruin3:getTerrainAbstractions(), 1)
      -- this other ruin does not block movement, and is far enough for the gap, thus not encroaching.
      lu.assertTrue(map:wouldTerrainFeatureBeValid(ruin3))
      
      -- now, give it a wall that is close
      ruinArgs.wallInfos = {{0,0,0,0,1,0,0,0}}
      local ruin4 = RuinFeature(ruinArgs)
      lu.assertEquals(#ruin4:getTerrainAbstractions(), 2)
      -- that close wall encroaches
      lu.assertFalse(map:wouldTerrainFeatureBeValid(ruin4))
      
      -- now, give it a wall that is far
      ruinArgs.wallInfos = {{0,0,0,0,0,0,1,1}}
      local ruin5 = RuinFeature(ruinArgs)
      -- that far wall is far enough that it does not encroach
      lu.assertTrue(map:wouldTerrainFeatureBeValid(ruin5))
      
    end

    function TestCarltographer:testRotationallySymmetricMap()
      local map = Map{mapOptions={xInches = 44, zInches = 30, isRotationallySymmetric = true}}
      
      -- Container in the middle is not ok
      local feature = ContainerFeature()
      lu.assertFalse(map:wouldTerrainFeatureBeValid(feature))
      
      -- Centerpiece is ok
      feature = ContainerFeature({isCenterpiece = true})
      lu.assertTrue(map:wouldTerrainFeatureBeValid(feature))
      
      -- off-center is ok, as it won't collide with its mirror.
      feature = ContainerFeature({isCenterpiece = true, transform={posZ=5, posX=8}})
      lu.assertTrue(map:wouldTerrainFeatureBeValid(feature))
      
      -- with the feature added, we can't something that would infringe on its mirror
      table.insert(map.terrainFeatures, feature:clone())
      feature = ContainerFeature({isCenterpiece = true, transform={posZ=-8, posX=-10}})
      lu.assertFalse(map:wouldTerrainFeatureBeValid(feature))
      
      -- but something that doesn't infringe on the mirror is ok
      feature = ContainerFeature({isCenterpiece = true, transform={posZ=-8, posX=3}})
      lu.assertTrue(map:wouldTerrainFeatureBeValid(feature))
    end
    
    function TestCarltographer:testRandomMapHappyPath()
      -- if true then return end  -- uncomment to skip this test
      math.randomseed(os.clock()*1000000)
      local ct = Carltographer({numGeneticIterations=10})
      ct.map = Map()
      ct:runGeneticAlgorithm(RealRandomGen())
    end
    
    function TestCarltographer:testSerializationOfContainerTerrain()
      local ct = Carltographer()
      ct.map = Map()
      lu.assertEquals(#ct.map.terrainFeatures, 0)
      local feature = ContainerFeature()
      table.insert(ct.map.terrainFeatures, feature:clone())
      lu.assertEquals(#ct.map.terrainFeatures, 1)
      
      local text = json.encode(ct)
      local ct2 = deserializeCarltographer(json.decode(text))
      lu.assertEquals(#ct2.map.terrainFeatures, 1)
      local feature2 = ct2.map.terrainFeatures[1]
      lu.assertTrue(feature2:is_a(TerrainFeature))
      lu.assertTrue(feature2:is_a(ContainerFeature))
      lu.assertFalse(feature2:is_a(CubeFeature))
    end

-- class TestCarltographer
-- LuaUnit:run()
lu.LuaUnit.verbosity = 2
os.exit(lu.LuaUnit.run())