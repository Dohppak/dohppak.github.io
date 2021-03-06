/* eslint-disable jsx-a11y/accessible-emoji */
import React from 'react';
import styled from '@emotion/styled';
import { Link, A } from './header-footer-anchor';
import cvLink from '../docs/SeungHeonDoh(May12).pdf';

const download = <a href={cvLink}>cv</a>;

const Header = styled.header(({ theme }) => ({
  padding: `0 ${theme.spacing}`,
  display: 'flex',
  flexDirection: 'row',
  justifyContent: 'space-between',
  height: theme.headerHeight,
  lineHeight: theme.headerHeight,
  color: theme.textColor,
  [theme.smallMedia]: {
    flexDirection: 'column',
    textAlign: 'center',
    justifyContent: 'space-between',
    height: '50px',
    lineHeight: '20px',
  },
  [theme.largeMedia]: {
    ...theme.centerPadding,
  },
}));

const H1 = styled.h1(({ theme }) => ({
  fontSize: '1.25rem',
  margin: 0,
  [theme.smallMedia]: {
    fontSize: '1rem',
  },
}));

const Small = styled.small({
  fontFamily: 'Times',
  fontWeight: 'bold',
  fontStyle: 'italic',
  fontSize: '75%',
  opacity: 0.35,
});

const Nav = styled.nav(({ theme }) => ({
  display: 'flex',
  flexDirection: 'row',
  fontWeight: 'normal',
  justifyContent: 'space-between',
  textTransform: 'uppercase',
  a: {
    marginLeft: theme.spacing,
    fontWeight: 'normal',
    fontSize: '0.8em',
  },
}));

const SiteHeader = () => (
  <Header>
    <H1>
      <Link to="/">SeungHeonDoh</Link>
      {' '}
      <Small>DeepLearning Researcher & Designer</Small>
    </H1>
    <Nav>
      <Link to="/">BLOG</Link>
      <Link to="/about">ABOUT</Link>
      <Link to="/contact">Contact</Link>
      {/* {download} */}
    </Nav>
  </Header>
);

export default SiteHeader;
