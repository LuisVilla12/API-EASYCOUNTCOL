DROP TABLE IF EXISTS `samples`;
CREATE TABLE `samples` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `sampleName` varchar(150) NOT NULL,
  `idUser` int(11) NOT NULL,
  `typeSample` varchar(255) NOT NULL,
  `volumenSample` varchar(255) NOT NULL,
  `factorSample` varchar(255) NOT NULL,
  `sampleRoute` varchar(255) NOT NULL,
  `creationDate` date NOT NULL,
  `processingTime` float NOT NULL,
  `count` int(4) NOT NULL,
  `creationTime` time DEFAULT NULL,
  `medioSample` varchar(255) NOT NULL,
  `state` int(1) DEFAULT NULL,
  UNIQUE KEY `id_sample` (`id`),
  KEY `fk_users` (`idUser`),
  CONSTRAINT `fk_users` FOREIGN KEY (`idUser`) REFERENCES `users` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=72 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

DROP TABLE IF EXISTS `users`;
CREATE TABLE `users` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(155) NOT NULL,
  `lastname` varchar(155) NOT NULL,
  `email` varchar(255) NOT NULL,
  `password` varchar(255) NOT NULL,
  `username` varchar(255) NOT NULL,
  `type` int(11) DEFAULT 1,
  PRIMARY KEY (`id`),
  UNIQUE KEY `id` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=19 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
