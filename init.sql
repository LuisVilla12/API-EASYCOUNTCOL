/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

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

INSERT INTO `samples` (`id`, `sampleName`, `idUser`, `typeSample`, `volumenSample`, `factorSample`, `sampleRoute`, `creationDate`, `processingTime`, `count`, `creationTime`, `medioSample`, `state`) VALUES
(42, 'Muestra 11', 3, 'Ambiental', '12ml', '1ml', '34071ecf39a54ee39badf1652633e11b_scaled_1000000033.jpg', '2025-04-27', 0.0618091, 15, '14:00:45', 'Agar nutritivo', 0),
(43, 'M1289', 3, 'Ambiental', '12ml', '1ml', '21903a399d0b4efda51b0df52636f6a0_scaled_cb286f34-67e8-4a16-a9e7-7667c92187771682348547686631350.jpg', '2025-04-27', 0.0192897, 110, '17:10:00', 'Agar nutritivo', 0),
(44, 'M12', 3, 'Ambiental', '1ml', '1ml', 'd91585d8fad14e5482eea7d06592e826_scaled_d7b65386-e75f-4181-a52d-dc339fc18ff1597125835736013386.jpg', '2025-04-27', 0.0437551, 210, '20:40:00', 'Agar nutritivo', 0),
(45, 'M12111', 3, 'Ambiental', '12ml', '1ml', 'd1be6b415a694761b2e2258370bac8da_scaled_515864ce-b362-4c0e-9082-6a6a310a85c65562940258754242449.jpg', '2025-04-27', 0.0185156, 70, '08:14:00', 'Agar nutritivo', 0),
(46, 'M1', 3, 'Alimentos', '1ml', '2ml', '119bd0ee5e1e40e7ade33a44a86cca1e_scaled_3ec294f6-df76-497c-98b5-96560fdb5b281137067492803920441.jpg', '2025-04-27', 0.0302548, 120, '12:24:00', 'Agar nutritivo', 0),
(47, 'Muestra 189', 3, 'Ambiental', '1ml', '1ml', '096c7dc6f99542aeb5d69e72cba68938_scaled_1000000033.jpg', '2025-04-27', 0.0122433, 10, '24:49:00', 'Agar nutritivo', 0),
(48, 'Prueba IOS', 3, 'Alimentos', '12ml', '12', 'dbd40d0c09f941eaa9535def802ef4cf_image_picker_761221C6-04DD-49E9-B623-127BFA9CA5C5-14510-000005CF1EDBD254.jpg', '2025-04-28', 0.676665, 78, '17:09:00', 'Agar nutritivo', 1),
(49, 'Muestra 1901', 3, 'Alimentos', '12ml', '1ml', 'fb392264a2eb460cb444ac5777830ec7_scaled_aacc2e0c-8578-473a-8397-68227950cc583573286966439028863.jpg', '2025-04-28', 0.0944822, 90, '16:11:00', 'Agar nutritivo', 1),
(50, 'Muestra 21', 3, 'Alimentos', '12ml', '1ml', 'f08e93ac8e9a4e82bfef896884afd3ac_scaled_c83c4433-25e8-41ca-beff-a07fb6e260eb6955350526912201442.jpg', '2025-04-28', 0.0432665, 78, '13:49:00', 'Agar nutritivo', 1),
(51, 'Muestra Z12', 18, 'Alimentos', '12ml', '1ml', '6581c228926b40d7a98e3f9df9b22811_scaled_04b90a0a-db97-424d-bddc-4eef92caecda8200079754385279693.jpg', '2025-04-29', 0.0900488, 178, '08:16:00', 'Agar nutritivo', 1),
(52, 'Muestra ', 18, 'Alimentos', '1vl', '1ml', '953ec281f2d04d95bd500dc46e1d53fa_scaled_68498ed8-5621-4949-a6bb-aebcfbc431414309522036046428919.jpg', '2025-05-01', 0.0924377, 1, '10:45:00', 'Agar nutritivo', 1),
(53, 'Muestra Nueva', 15, 'Alimentos', '1ml', '2ml', '40a30fab8df745778adc2ff4d53a4c26_scaled_1000000033.jpg', '2025-05-01', 0.0500596, 1, '19:06:00', 'Agar nutritivo', 1),
(54, 'Muestra 1989', 18, 'Material', '1ml', '1ml', 'ea1aa4648b2f44a3b50011823a2e9fbd_scaled_e3c87256-9109-4bf1-8078-974954d6ed838149944825476007679.jpg', '2025-05-03', 0.0812888, 1, '12:10:00', 'Agar nutritivo', 1),
(55, 'Muestra 1211', 18, 'Material', '1ml', '1ml', '088a56287a6f4585ab2eefb4fcf575d4_scaled_b77e7995-7442-4e75-8c21-85aa26e5562b1070691026061908156.jpg', '2025-05-06', 0.0766313, 5, '18:19:00', 'Agar nutritivo', 1),
(56, 'Muestra 123123', 18, 'Material', '2ml', '1ml', 'e590b19167d846d69b5bfa5079517b64_scaled_7f5d5c5a-306a-4c4a-9fa7-36a4c7fc1f7d6439678124377128051.jpg', '2025-05-07', 0.089396, 5, '16:49:00', 'Agar nutritivo', 1),
(57, 'asd', 18, 'Material', 'asdfasd', 'dafasdf', 'a918ae128bad445db733a69994ff22a4_scaled_712911ac-cef1-461a-b519-ce76bb1a2e4e5807255094300834678.jpg', '2025-05-08', 0.0560133, 5, '11:00:00', 'Agar nutritivo', 1),
(58, 'adsfasd', 18, 'Material', 'asdfasdf', 'asdfasdf', '508d0698c0014b0988378b59c081114c_scaled_1e88a951-b591-40bb-add1-d1b38dddf5f87122346933652542256.jpg', '2025-05-09', 0.0585933, 5, '09:43:00', 'Agar nutritivo', 1),
(59, 'asdadsf', 18, 'Otras muestras', 'adsfasdfasd', 'dsfasdfda', '4c8caedae2ae40899031e0f33003c8fb_scaled_2cf25495-4fbb-4fd5-8782-c1253b0f38d05242133028001671892.jpg', '2025-05-10', 0.0901525, 5, '19:45:00', 'Agar nutritivo', 1),
(60, 'Muestra 1', 18, 'Otras muestras', '1ml', '1ml', '061ba47347de42cc9388c1f1f2fe3967_scaled_623f0f1c-a15a-40d3-8ed8-73bb5f50aced509241551119331205.jpg', '2025-05-11', 0.100207, 5, '08:15:00', 'Agar nutritivo', 1),
(61, 'AAAA', 18, 'Otras muestras', 'CCCCC', 'DDDD', '430e73fb0ce8479bad3878c9d3d54771_scaled_5243ca69-d155-4b10-b0af-05edaa9db5c76122169426819245945.jpg', '2025-05-12', 0.0974619, 5, '14:50:01', 'Agar nutritivo', 1),
(62, 'Luis', 18, 'Otras muestras', '1ml', '1ml', '0346232e82b34dc3a1ad06e449c11664_scaled_22940d9e-93f9-412f-914a-5916822cd06f4392296766981850922.jpg', '2025-06-12', 0.191738, 5, '18:02:45', 'Agar nutritivo', 1),
(63, 'Luis2', 18, 'Otras muestras', '1ml', '1ml', '0346232e82b34dc3a1ad06e449c11664_scaled_22940d9e-93f9-412f-914a-5916822cd06f4392296766981850922.jpg', '2025-06-13', 0.191738, 5, '16:42:00', 'Agar nutritivo', 1),
(64, '65', 18, 'Clinica - Biológica', '1ml', '1ml', '0346232e82b34dc3a1ad06e449c11664_scaled_22940d9e-93f9-412f-914a-5916822cd06f4392296766981850922.jpg', '2025-07-13', 0.191738, 5, '22:10:00', 'Agar nutritivo', 1),
(65, 'Luis2', 18, 'Clinica - Biológica', '1ml', '1ml', '0346232e82b34dc3a1ad06e449c11664_scaled_22940d9e-93f9-412f-914a-5916822cd06f4392296766981850922.jpg', '2025-07-18', 0.191738, 5, '07:16:00', 'Agar nutritivo', 1),
(66, 'Luis2', 18, 'Clinica - Biológica', '1ml', '1ml', '0346232e82b34dc3a1ad06e449c11664_scaled_22940d9e-93f9-412f-914a-5916822cd06f4392296766981850922.jpg', '2025-05-04', 0.191738, 5, '23:45:00', 'Agar nutritivo', 1),
(67, 'Muestra TEST', 1, 'Ambiental', '1', '1', 'f68b8938949e4861921a27e80859f96d_CAP2930747516470384681.jpg', '2025-07-28', 0.0365634, 5, '22:06:06', 'Agar nutritivo', 1),
(68, 'Luis2', 18, 'Clinica - Biológica', '1ml', '1ml', '0346232e82b34dc3a1ad06e449c11664_scaled_22940d9e-93f9-412f-914a-5916822cd06f4392296766981850922.jpg', '2025-05-04', 0.191738, 5, '18:02:45', 'Agar nutritivo', 1),
(69, 'Muestra UFC BIEN', 1, 'Ambiental', '11', '11', '89800c8aa99b4586b5640dfa3dd48a94_scaled_imagen-test2.jpg', '2025-08-22', 14.3671, 33, '12:14:37', 'Agar MacConkey', NULL),
(70, 'UFC BIEN', 16, 'Ambiental', '11', '11', '8a30b784789844ab8cba2df08ca3a875_scaled_imagen-test3.jpg', '2025-08-22', 30.9024, 10, '12:42:45', 'Agar nutritivo', NULL),
(71, 'Muestra UFC', 16, 'Alimentos', '1', '1', '9ab43e8309114fa99356020ba8b0e2a4_scaled_36.jpg', '2025-08-22', 20.7455, 33, '15:54:51', 'Agar MacConkey', NULL);
INSERT INTO `users` (`id`, `name`, `lastname`, `email`, `password`, `username`, `type`) VALUES
(1, 'Luis Alberto', 'Jimenez Villa', 'luisjivl.01@gmail.com', '$2b$12$1fkSbEIEpqsAn88bODWGCukEhp0Daj79W14Rkykshl9Z.Kjg4Yw5e', 'LuisJi', 1),
(3, 'Rosaa', 'Villa Reyes', 'rosa@gmail.com', '$2b$12$1fkSbEIEpqsAn88bODWGCukEhp0Daj79W14Rkykshl9Z.Kjg4Yw5e', 'VillaRosa', 1),
(4, 'Juan Carlos', 'Perez Garcia', 'asdfadsf@gmail.com', '$2b$12$opS4L.ZXmEA3O9ngThslYOwWPhcSZjU9mZeSgSu8YZro2eQw/jvLm', 'Juan4878', 1),
(6, 'Rosa', 'Villa Reyes', 'rosavilla12@gmail.com', '$2b$12$qnp7U5eanbkKxBoBWygQpensSdSZSzVBWcihYs3MwYoZ.d9b8Gm96', 'RosaVilla12', 1),
(10, 'Hector', 'Jimenez Villa', 'hector@gmail.com', '$2b$12$teheWK7elIU..ay/V0JM.uJ1.HKz7X/6f.S29c4aA1toqa2LcKRBa', 'Hector05', 1),
(14, 'Raul', 'Jimenez ', 'raul@gmail.com', '$2b$12$3N.WFL4uIPMIQE8VZQdeFuQHEXIQXbojYJ7Ioz5CctogjxPqqH9MO', 'Raul11', 1),
(15, 'Nancy Elizabeth', 'Laguna Rivera', 'nancy@gmail.com', '$2b$12$7Uy7qnzlgMJRDh.oIr2d..2YbBroUhhxieYq3ncWxwu7bnD7aj13u', 'Nancy23', 1),
(16, 'Prueba', 'prueba ', 'luis@gmail.com', '$2b$12$kYrzyIXP1BAX8LWQLQwx9.maMzZWsi8oagOU5.30t4lSFmLy30ve.', 'prueba', 1),
(17, 'Edgar', 'Lopez Garcia', 'edgar@gmail.com', '$2b$12$Scc8W8JTjSvos.qu6/deNuYhBkvHtJjT/imbg9U0dAItiTEjptZee', 'EdgarGarcia ', 1),
(18, 'Luis ', 'Villa', 'admin@gmail.com', '$2b$12$/K.a148eFyTBT8zes1Gvz.qq5pWxRbDGs9lq1PQHaxPrUiktjT6GS', 'luisvilla', 1);


/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;