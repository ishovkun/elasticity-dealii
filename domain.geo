cl1 = 1;
xSize = 10;
ySize = 10;

Point(1) = {-xSize/2, -ySize/2, 0, 1};
Point(2) = {+xSize/2, -ySize/2, 0, 1};
Point(3) = {+xSize/2, +ySize/2, 0, 1};
Point(4) = {-xSize/2, +ySize/2, 0, 1};


// lines of the outer box:
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

// loops of the outside and the two cutouts
Line Loop(5) = {1, 2, 3, 4};
Plane Surface(6) = {5};

// these define the boundary indicators in deal.II:
Physical Line(0) = {1}; // Bottom
Physical Line(1) = {2}; // Right
Physical Line(2) = {3}; // Top
Physical Line(3) = {4}; // Left

// you need the physical surface, because that is what deal.II reads in
Physical Surface(7) = {6};

// some parameters for the meshing:
Mesh.Algorithm = 8;
Mesh.RecombineAll = 1; // to get quadrelaterals
// Mesh.CharacteristicLengthFactor = 0.09;
Mesh.SubdivisionAlgorithm = 2;
Mesh.Smoothing = 20;
Show "*";


///